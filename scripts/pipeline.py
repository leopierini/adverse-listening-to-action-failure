import sys
import json
import os
import re
import base64
import io
import threading

# Force HuggingFace to save all downloaded models in the project "models" folder
# (scripts/ live one level below the project root → go up one dir).
os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

import httpx
import mlx_whisper
from openai import OpenAI

# ─── LLM backend (OpenAI-compatible HTTP) ─────────────────────────────────────
# All three routers speak the OpenAI API: Qwen3.5-9B locally via `mlx_lm.server`,
# Qwen3.5-27B and Gemma 4 31B on the GCP A100 via vLLM. We talk to all of them
# through one client, parameterised by base_url + model name.
# Override via env (LLM_BASE_URL / LLM_MODEL) or per-call arguments.
DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/Qwen3.5-9B-MLX-8bit")
DEFAULT_LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")  # local servers ignore it
# Thinking is OFF by default: this is an intent *router*, not a reasoning task,
# so we keep the three models comparable and the sweep cheap.
DEFAULT_ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"

# A routing answer is one small JSON object. Capping the budget stops a runaway
# generation (e.g. a model whose thinking mode we failed to disable) from burning
# billed A100 minutes on thousands of tokens per request.
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "256"))

# Timeouts (audit 2026-08-12): the SDK defaults are read=600s with max_retries=2,
# i.e. one hung request pins a worker for ~30 minutes of billed GPU. The sidecar
# plus resume already handle retries, so we retry once and fail fast.
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 120.0
MAX_RETRIES = 1

_clients = {}
_clients_lock = threading.Lock()


def _get_client(base_url):
    """One OpenAI client per endpoint, cached so we don't rebuild it per call
    (connections are pooled and kept alive across the 16-32 worker threads)."""
    with _clients_lock:
        if base_url not in _clients:
            _clients[base_url] = OpenAI(
                base_url=base_url,
                api_key=DEFAULT_LLM_API_KEY,
                timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                max_retries=MAX_RETRIES,
            )
        return _clients[base_url]


# ─── Endpoint verification ────────────────────────────────────────────────────
# `llm_model` is what the ENTIRE H2a/H2b analysis groups on, and until now it
# recorded what the client *asked for*, never what the server actually loaded.
# Forgetting --llm-base-url while the local mlx_lm.server is still up on :8080
# would route every request through Qwen-9B and stamp it "Qwen3.5-27B", making
# H2a a comparison of Qwen-9B with itself — invisibly. One GET closes that.

class ModelNotServedError(RuntimeError):
    """The endpoint does not serve the model we were about to record."""


def fetch_served_models(base_url):
    """Model ids the endpoint reports serving (GET /v1/models)."""
    return [m.id for m in _get_client(base_url).models.list().data]


def verify_model_served(model, base_url):
    """Confirm `base_url` really serves `model`. Returns the served id.
    Raises ModelNotServedError — callers should abort the run, never continue."""
    try:
        served = fetch_served_models(base_url)
    except Exception as e:
        raise ModelNotServedError(
            f"Could not list models at {base_url}: {e!r}\n"
            f"  Is the server up, and is --llm-base-url correct?"
        ) from e
    if model not in served:
        raise ModelNotServedError(
            f"Endpoint {base_url} does not serve {model!r}.\n"
            f"  It serves: {served}\n"
            f"  Either --llm-base-url is wrong (still pointing at the local\n"
            f"  mlx_lm.server?) or --llm-model is. Refusing to run: every row\n"
            f"  would record the wrong llm_model and the sweep would be worthless."
        )
    return model


# ─── Thinking mode ────────────────────────────────────────────────────────────

def thinking_request(model=None, enable_thinking=None):
    """The extra_body we send to request thinking-OFF, for this model's family.

    Qwen (vLLM and mlx_lm.server) honours `chat_template_kwargs.enable_thinking`.

    ⚠️ UNRESOLVED for Gemma 4 (audit 2026-08-12): its switch could not be
    determined from this repo or any installed package (no vLLM, no Gemma
    weights, no Gemma chat template on disk). vLLM passes unknown
    chat_template_kwargs into the Jinja render where they are silently unused —
    so for Gemma this is very likely a NO-OP and thinking is NOT off, contrary
    to tracker §4/§5/§11. We deliberately do not guess a second key.

    Instead: whatever we send is recorded on every row as `llm_thinking_request`,
    and `--dump-raw-responses N` saves the first N raw responses per model so the
    Phase-7 pilot can VERIFY thinking is off rather than assume it. Resolve the
    Gemma convention against the pinned vLLM build before the Gemma sweep."""
    if enable_thinking is None:
        enable_thinking = DEFAULT_ENABLE_THINKING
    if enable_thinking:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


# ─── Raw-response capture (pilot inspection) ──────────────────────────────────

_dump_lock = threading.Lock()
_dump_state = {"path": None, "limit": 0, "counts": {}}


def enable_raw_dump(path, limit):
    """Save the first `limit` raw responses per model to `path` (JSONL).
    Used by the Phase-7 pilot to eyeball thinking-mode and JSON compliance."""
    with _dump_lock:
        _dump_state["path"] = path
        _dump_state["limit"] = int(limit)
        _dump_state["counts"] = {}


def _dump_raw(model, base_url, content, finish_reason, has_reasoning, extra_body):
    """Record a raw response before it is validated or parsed, so the pilot sees
    the unusable ones too (that is the whole point)."""
    if not _dump_state["path"] or _dump_state["limit"] <= 0:
        return
    with _dump_lock:
        n = _dump_state["counts"].get(model, 0)
        if n >= _dump_state["limit"]:
            return
        _dump_state["counts"][model] = n + 1
        rec = {
            "model": model, "base_url": base_url, "finish_reason": finish_reason,
            "reasoning_content_present": has_reasoning, "extra_body": extra_body,
            "content": content,
        }
        try:
            with open(_dump_state["path"], "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ [raw-dump] could not write {_dump_state['path']}: {e!r}")


# ─── Response validation ──────────────────────────────────────────────────────

def _usable_content(response, model, base_url, extra_body):
    """Return the assistant text, or None if the response is unusable.

    Audit 2026-08-12 (finding C3): an HTTP 200 carrying empty or truncated
    content used to fall through to the JSON parser, become "JSON_PARSE_ERROR"
    and be SCORED as tsa=0 — fabricating a wrong answer the model never gave.
    The two live triggers on this stack:
      - vLLM started with a reasoning parser leaves `content` None and puts
        everything in `reasoning_content` → every row would score tsa=0;
      - finish_reason == "length" → the answer was cut off (a resource problem,
        not a model mistake).
    Both are infrastructure failures: they belong in the sidecar and get retried."""
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    finish_reason = getattr(choice, "finish_reason", None)
    has_reasoning = bool(getattr(message, "reasoning_content", None))

    _dump_raw(model, base_url, content, finish_reason, has_reasoning, extra_body)

    if not content or not content.strip() or finish_reason != "stop":
        return None, (
            f"unusable response: finish_reason={finish_reason!r}, "
            f"content_chars={len(content or '')}, "
            f"reasoning_content_present={has_reasoning}"
        )
    return content, None


def model_family_of(model_id):
    """Map a model id → experimental family ('qwen' | 'google' | None).
    Used to stamp `model_family` on every row so the H4 pathway×family design
    can be reconstructed from the JSONL alone."""
    if not model_id:
        return None
    m = model_id.lower()
    if "qwen" in m:
        return "qwen"
    if "gemma" in m or "google" in m:
        return "google"
    return None


def transcribe_audio(audio_path, model_path="mlx-community/whisper-large-v3-turbo", language="en"):
    print(f"🎙️ [ASR] Transcribing {audio_path} with {model_path}...")
    # Note: MLX downloads it from HuggingFace on the first run
    # language="en" forces English transcription — prevents Whisper from auto-switching
    # languages and hallucinating, e.g., Polish on English audio (see Sample #1 incident).
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_path,
        language=language,
    )
    transcription = result["text"].strip()
    print(f"📝 [ASR Output] {transcription}")
    return transcription

# The primary 9-intent, 9-shot routing prompt ("primary_9intent_9shot").
# Module-level so prompt_ablation.py can reuse/extend it without duplicating.
SYSTEM_PROMPT = """You are an Intent Router for a voice assistant. You classify transcriptions into EXACTLY one of these intent labels:

ALLOWED INTENTS (use these EXACT strings, copy-paste them):
- "calendar_set"
- "calendar_query"
- "calendar_remove"
- "alarm_set"
- "alarm_query"
- "alarm_remove"
- "lists_createoradd"
- "lists_query"
- "lists_remove"
- "unknown"

WRONG labels (DO NOT use these): "set_alarm", "schedule_meeting", "set_calendar", "check_calendar", "send_message", "add_to_list", "create_list", "remove_from_list", "check_list", "qa", "get_info"

You must respond ONLY with valid JSON. No markdown, no explanation.
Extract relevant entities into 'parameters' (time, date, person, event_name, item_name, list_name).
If unsure, use "unknown".

Example 1:
User: "Schedule a meeting with John at 3pm"
{"intent": "calendar_set", "parameters": {"time": "3:00 PM", "person": "John"}}

Example 2:
User: "Wake me up at 6 in the morning"
{"intent": "alarm_set", "parameters": {"time": "6:00 AM"}}

Example 3:
User: "What's on my calendar tomorrow?"
{"intent": "calendar_query", "parameters": {"date": "tomorrow"}}

Example 4:
User: "Cancel my 2 o'clock alarm"
{"intent": "alarm_remove", "parameters": {"time": "2:00 PM"}}

Example 5:
User: "Add milk to my grocery list"
{"intent": "lists_createoradd", "parameters": {"item_name": "milk", "list_name": "grocery"}}

Example 6:
User: "What's on my shopping list?"
{"intent": "lists_query", "parameters": {"list_name": "shopping"}}

Example 7:
User: "Remove eggs from the list"
{"intent": "lists_remove", "parameters": {"item_name": "eggs"}}

Example 8:
User: "Clear my schedule for today"
{"intent": "calendar_remove", "parameters": {"date": "today"}}

Example 9:
User: "Wipe out all my alarms"
{"intent": "alarm_remove", "parameters": {}}"""


def get_intent_from_llm(transcription, model=None, base_url=None, enable_thinking=None,
                        system_prompt=None):
    model = model or DEFAULT_LLM_MODEL
    base_url = base_url or DEFAULT_LLM_BASE_URL
    if enable_thinking is None:
        enable_thinking = DEFAULT_ENABLE_THINKING
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    print(f"🧠 [LLM] Routing intent with {model} @ {base_url} ...")

    client = _get_client(base_url)

    # Disable the model's "thinking" mode for routing — see thinking_request().
    extra_body = thinking_request(model, enable_thinking)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription},
            ],
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format={"type": "json_object"},
            extra_body=extra_body or None,
        )
        output, problem = _usable_content(response, model, base_url, extra_body)
    except Exception as e:
        print(f"⚠️ [Error] LLM request failed: {e!r}")
        return {"error": "llm_request_failed", "detail": repr(e)}

    if problem is not None:
        print(f"⚠️ [Error] {problem}")
        return {"error": "llm_request_failed", "detail": problem}

    print(f"🤖 [LLM Output Raw]\n{output}\n")
    return _parse_json_output(output)

def flac_to_wav_b64(audio_path):
    """Read an audio file (FLAC/WAV) and return base64-encoded mono WAV at the
    file's NATIVE sample rate (no resampling — the entire dataset + corrupted
    bank is 16 kHz by construction, TARGET_SAMPLE_RATE in corrupt_audio.py).
    The omni endpoints (vLLM OpenAI-compatible) take a base64 WAV content part;
    the corrupted bank is FLAC, so we transcode in-memory (no second bank on disk)."""
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)  # force mono
    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_omni_messages(audio_b64, system_prompt, fmt="wav"):
    """OpenAI-style messages with an input_audio user turn. The system prompt is
    the SAME 9-intent/9-shot text the cascade routers get (a deliberate design
    constant; the few-shot exemplars are text, the query is audio — see tracker §11)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}},
        ]},
    ]


def _parse_json_output(output):
    """Shared JSON extraction: strip markdown fences, else first {...} block."""
    clean = (output or "").strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"error": "Invalid JSON format", "raw_output": output}


def get_intent_from_omni(audio_path, model=None, base_url=None, enable_thinking=None,
                         system_prompt=None):
    """End-to-end omni routing: audio in → action JSON out, NO transcript.
    Same backend/return contract as get_intent_from_llm (parsed dict or {"error":...})."""
    model = model or DEFAULT_LLM_MODEL
    base_url = base_url or DEFAULT_LLM_BASE_URL
    if enable_thinking is None:
        enable_thinking = DEFAULT_ENABLE_THINKING
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    print(f"🔊 [OMNI] Routing intent from audio with {model} @ {base_url} ...")

    client = _get_client(base_url)
    extra_body = thinking_request(model, enable_thinking)

    try:
        audio_b64 = flac_to_wav_b64(audio_path)
        messages = build_omni_messages(audio_b64, system_prompt)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format={"type": "json_object"},
            extra_body=extra_body or None,
        )
        output, problem = _usable_content(response, model, base_url, extra_body)
    except Exception as e:
        print(f"⚠️ [Error] Omni request failed: {e!r}")
        return {"error": "llm_request_failed", "detail": repr(e)}

    if problem is not None:
        print(f"⚠️ [Error] {problem}")
        return {"error": "llm_request_failed", "detail": problem}

    print(f"🤖 [OMNI Output Raw]\n{output}\n")
    return _parse_json_output(output)


def process_pipeline(audio_path, asr_model_path="mlx-community/whisper-large-v3-turbo"):
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file '{audio_path}' not found.")
        return
        
    transcription = transcribe_audio(audio_path, model_path=asr_model_path)
    result = get_intent_from_llm(transcription)
    
    print("✅ [Final Pipeline Result]")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_audio_file> [asr_model_hf_path]")
        print("Example: python pipeline.py test_audio/sample.wav mlx-community/whisper-base")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    asr_model = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/whisper-large-v3-turbo"
    process_pipeline(audio_file, asr_model)
