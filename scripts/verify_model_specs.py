#!/usr/bin/env python
"""Check every model the thesis plans to serve against reality.

WHY THIS EXISTS
---------------
On 2026-08-14 the model specs in docs/design/STACK.md §5/§11 and the session-1
runbook were checked against Hugging Face for the first time. Two of the four
repo ids named the BASE model where the design needs the instruction-tuned one
(`google/gemma-4-31b` vs `google/gemma-4-31b-it`), and the omni checkpoint's
size was recorded as "~20 GB" when the real weights total 27.6 GB -- a figure
that had propagated to four places in the tracker and become a risk rating
("A100 40 GB OOM: Low").

None of that was a new mistake. It was months-old prose that had never been
checked once. This script makes "the model names are right" a thing you run in
ten seconds instead of a thing you hope.

It verifies, per model:
  * the repo resolves (HTTP 200 / gated / absent)
  * whether it is instruction-tuned  (chat template, `conversational` tag,
    `base_model:finetune:` provenance)
  * the real weight size, summed from the actual .safetensors files
  * whether it fits the target VRAM with room for the KV cache
  * quantization format vs the GPU's compute capability (FP8 needs Ada/Hopper;
    an A100 is Ampere and cannot accelerate it)

Usage:
    ./venv/bin/python scripts/verify_model_specs.py
    ./venv/bin/python scripts/verify_model_specs.py --vram 80

Exit 0 = every model checks out. 1 = at least one problem.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

HF_API = "https://huggingface.co/api/models/"

# The A100 40 GB in instance-pierini-gpu. Ampere = compute capability 8.0.
DEFAULT_VRAM_GB = 40.0
# vLLM needs room beyond the weights: CUDA context, activations, and the KV
# cache. Below this margin the server will not start on realistic batch sizes.
MIN_HEADROOM_GB = 6.0

# Formats an Ampere card cannot accelerate, mapped to why.
BAD_ON_AMPERE = {
    "fp8": "FP8 needs Ada (8.9) or Hopper (9.0); A100 is Ampere (8.0)",
    "gguf": "GGUF targets llama.cpp; vLLM support is experimental",
}

# What the design needs each role to be. `instruct` = must follow a 9-shot
# prompt and emit JSON, so a base checkpoint is disqualifying.
MODELS = [
    {"role": "cascade router, large (H2a)",
     "id": "Qwen/Qwen3.5-27B-GPTQ-Int4", "instruct": True},
    {"role": "cascade router, cross-family (H2b)",
     "id": "google/gemma-4-31B-it-qat-w4a16-ct", "instruct": True},
    {"role": "omni, Qwen pathway (H4)",
     "id": "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit", "instruct": True},
    {"role": "omni, Google pathway (H4)",
     "id": "google/gemma-4-12B-it-qat-w4a16-ct", "instruct": True},
]

# Un-suffixed ids that look right and are wrong. Checked too, so the failure
# mode that cost this project a day is caught rather than rediscovered:
# the first two are BASE checkpoints (no chat template -> cannot follow the
# 9-shot JSON prompt); the last two are BF16 and do not fit a 40 GB card.
TRAPS = [
    ("google/gemma-4-31b", "BASE checkpoint — use google/gemma-4-31B-it-qat-w4a16-ct"),
    ("google/gemma-4-12b", "BASE checkpoint — use google/gemma-4-12B-it-qat-w4a16-ct"),
    ("Qwen/Qwen3.5-27B", "BF16 55.6 GB — use Qwen/Qwen3.5-27B-GPTQ-Int4"),
    ("google/gemma-4-31B-it", "BF16 62.5 GB — use the -qat-w4a16-ct build"),
    ("Qwen/Qwen3.5-27B-FP8", "FP8 needs Ada/Hopper; the A100 is Ampere (8.0)"),
]


def fetch(model_id):
    """Return (status, payload). status in ok|gated|absent|error."""
    req = urllib.request.Request(HF_API + model_id,
                                 headers={"User-Agent": "thesis-preflight"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return "ok", json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return "gated", None
        if e.code == 404:
            return "absent", None
        return "error", "HTTP {}".format(e.code)
    except Exception as e:  # network, timeout, JSON
        return "error", str(e)[:120]


def weight_gb(model_id):
    """Sum the real .safetensors sizes. Returns None if the tree is unreadable."""
    url = HF_API + model_id + "/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-preflight"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            tree = json.loads(r.read().decode())
    except Exception:
        return None
    total = sum(f.get("size", 0) for f in tree
                if f.get("path", "").endswith(".safetensors"))
    return total / 1e9 if total else None


def is_instruct(payload):
    """Evidence that this is the instruction-tuned checkpoint, not the base."""
    tags = payload.get("tags", [])
    reasons = []
    if "conversational" in tags:
        reasons.append("tag `conversational`")
    if any(t.startswith("base_model:finetune:") for t in tags):
        reasons.append("declares base_model:finetune:")
    files = [f.get("rfilename", "") for f in payload.get("siblings", [])]
    if "chat_template.jinja" in files:
        reasons.append("ships a chat template")
    return (len(reasons) > 0), reasons


def quant_note(payload):
    tags = [t.lower() for t in payload.get("tags", [])]
    ident = (payload.get("modelId") or "").lower()
    for bad, why in BAD_ON_AMPERE.items():
        if bad in ident or bad in tags:
            return why
    return None


def main():
    ap = argparse.ArgumentParser(description="Verify the thesis model specs against Hugging Face.")
    ap.add_argument("--vram", type=float, default=DEFAULT_VRAM_GB,
                    help="target GPU VRAM in GB (default {})".format(DEFAULT_VRAM_GB))
    args = ap.parse_args()

    print("Verifying {} models against {:.0f} GB VRAM\n".format(len(MODELS), args.vram))
    problems = []

    for m in MODELS:
        mid = m["id"]
        print("=" * 68)
        print("{}\n  {}".format(mid, m["role"]))
        status, payload = fetch(mid)

        if status != "ok":
            print("  REPO       : {}".format(status.upper()))
            problems.append("{}: repo {}".format(mid, status))
            continue
        print("  repo       : resolves (canonical id: {})".format(
            payload.get("modelId") or payload.get("id")))

        if m["instruct"]:
            ok, reasons = is_instruct(payload)
            if ok:
                print("  variante   : instruction-tuned — {}".format(", ".join(reasons)))
            else:
                print("  variante   : NO EVIDENCE of instruction tuning — likely the BASE model")
                problems.append("{}: looks like a base checkpoint".format(mid))

        gb = weight_gb(mid)
        if gb is None:
            print("  peso       : non leggibile")
            problems.append("{}: size unreadable".format(mid))
        else:
            head = args.vram - gb
            verdict = "ok" if head >= MIN_HEADROOM_GB else "TROPPO GRANDE"
            print("  peso reale : {:.1f} GB  -> {:.1f} GB liberi per KV cache  [{}]"
                  .format(gb, head, verdict))
            if head < MIN_HEADROOM_GB:
                problems.append("{}: {:.1f} GB leaves only {:.1f} GB headroom"
                                .format(mid, gb, head))

        q = quant_note(payload)
        if q:
            print("  quantizz.  : ATTENZIONE — {}".format(q))
            problems.append("{}: {}".format(mid, q))

    print("=" * 68)
    print("Ids that look right and are WRONG — verifying each really is wrong:")
    for mid, why in TRAPS:
        # These were merely PRINTED until 2026-08-15, while the docstring said
        # "Checked too". An invented repo id sat in the list and the script
        # still exited 0. Now each trap is resolved and its claim tested.
        st, payload = fetch(mid)
        if st != "ok":
            print("  [{:6s}] {:44s} {}".format(st.upper(), mid, why))
            problems.append("TRAP {} does not resolve ({}) — the warning is stale"
                            .format(mid, st))
            continue
        gb = weight_gb(mid)
        ok_instr, _ = is_instruct(payload)
        bad_quant = quant_note(payload)
        # A trap is legitimate if it is a base checkpoint, or too big, or a
        # format this GPU cannot use. If none of those hold, the warning is
        # wrong and would send someone away from a usable model.
        justified = (not ok_instr) or (gb is not None and args.vram - gb < MIN_HEADROOM_GB) \
            or bad_quant is not None or "fp8" in mid.lower()
        print("  [{}] {:44s} {}".format("ok " if justified else "BAD", mid, why))
        if not justified:
            problems.append("TRAP {} is actually fine — the warning is wrong".format(mid))

    print("=" * 68)
    if problems:
        print("\n{} PROBLEMA/I:".format(len(problems)))
        for p in problems:
            print("  - {}".format(p))
        print("\nNon scrivere questi id in un runbook finche' non sono risolti.")
        return 1
    print("\nTutti i modelli verificati: repo reale, variante istruita, entra in VRAM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
