# From Adverse Listening to Action Failure

**Error propagation and robustness in LLM-based voice AI agents.**
Code, tests and analysis outputs for a master's thesis in *Psicologia Cognitiva Applicata*,
Università degli Studi di Padova.

Candidate: **Leonardo Pierini** · Supervisor: **Prof. Alberto Testolin**

---

## What the study does

A voice assistant does not just transcribe speech — it turns a spoken command into an
**executable action**: a structured function call with a label and parameters
(`alarm_set(time="07:00")`). That chain can be built two ways:

- **cascade** — speech recognition (ASR) produces a **transcript**, then a language model
  (LLM) reads the text and picks the action. Modular by construction: the ASR decides, and
  the LLM cannot revise the acoustic decision.
- **end-to-end (omni)** — one audio-native model consumes the speech directly and emits the
  action, with **no intermediate transcript**.

Three literatures meet here and stop at the same edge: ASR-robustness work measures
transcription error and stops at the text; LLM tool-calling work measures wrong or fabricated
calls but assumes clean text; cascade-vs-end-to-end work uses synthesised audio and
classification tasks. Nobody has measured the whole chain — physical degradation →
transcription error → action failure — or compared the two architectures **within the same
model family** on real recordings doing tool-calling.

The framing is psychological rather than engineering: the six degradations are categories of
*adverse listening condition* in the sense of Mattys, Davis, Bradlow & Scott (2012), and the
hypotheses descend from the psycholinguistic literature on phonemic restoration, lexical
constraints and the modularity-vs-feedback debate in speech perception.

## The design

416 **real** human recordings from the SLURP benchmark (calendar / alarm / lists scenarios,
headset microphone only), manually reviewed: 4 intent labels corrected, 5 utterances removed,
22 in-domain ambiguous utterances kept and flagged. Nine intent labels, six critical slot
types.

| degradation | mild / moderate / severe | category (Mattys, 2012) |
|---|---|---|
| additive noise | 20 / 10 / 0 dB SNR | transmission, **energetic masking** |
| babble (NOISEX-92, 100 talkers) | 20 / 10 / 0 dB SNR | energetic **+ informational** masking |
| reverberation | T60 0.3 / 0.6 / 1.2 s | transmission, no competing signal |
| Opus codec | 32 / 16 / 8 kbit/s | transmission, no competing signal |
| clipping | 50 / 25 / 10 % of peak | transmission, no competing signal |
| far-field | low-pass 6000 / 4000 / 2500 Hz + attenuation | transmission, no competing signal |

`416 utterances × 6 degradations × 3 severities = 7,488 audio files`, plus the clean
condition → **19 acoustic conditions**. Generation is deterministic: each file's RNG seed is
derived from its own name via SHA-256 (`scripts/corrupt_audio.py`), so rebuilding the bank
reproduces it byte for byte.

**Models — one per experimental *role*:**

| role | model |
|---|---|
| ASR reference | Whisper Large-v3 Turbo |
| ASR, different architecture family | Parakeet TDT 0.6B v3 (Token-and-Duration Transducer) |
| ASR, edge-deployment witness | Whisper Base |
| LLM router, small / edge (local, MLX 8-bit) | Qwen3.5-9B |
| LLM router, large — isolates **scale** | Qwen3.5-27B |
| LLM router, different vendor — external validity | Gemma 4 31B-it |
| omni, Qwen family | Qwen3-Omni-30B-A3B |
| omni, Google family | Gemma 4 12B-it |

All with thinking **off**, temperature 0, JSON output and the **same system prompt** — text in
the cascade, audio in the omni arm, so the architectural contrast is not contaminated by a
difference in instructions.

`171 cascade cells (3 ASR × 3 routers × 19 conditions) + 38 omni cells (2 × 19) = 209 cells
× 416 utterances = 86,944 inferences`, all collected.

**Measures, layered.** `TSA` (intent correct, 0/1) · `PF` (fraction of gold parameters
recovered — value-based soft matching, not key equality, because the model and the benchmark
use different slot vocabularies) · **`EES` = action succeeded = `TSA == 1 AND PF >= 0.5`**,
the primary outcome. `ees_strict` (`PF == 1.0`) is computed alongside as a robustness column.
Cascade rows additionally carry WER/CER and a slot-restricted error rate; these are
**undefined for the omni arm**, which produces no transcript.

**Confirmatory estimator, pinned before the data were seen:** GEE (generalized estimating
equations), binomial family, exchangeable working correlation clustered on the utterance
(416 clusters), robust standard errors; `PF` as a fractional logit. There is **no fallback**:
rank deficiency, perfect separation and non-finite standard errors each refuse the fit rather
than print a number that looks like a result.

## What is in this repository

```
scripts/           38 scripts: degradation, ASR, routing, scoring, analysis, figures
tests/             the test suite (440 tests)
results/analysis/  analysis outputs (model fits, absorption tables, mediation)
results/figures/   the 7 official figures, PDF and PNG
dataset/           labels, manifests and the manual-review logs
docs/gcp-session-*/  the serving scripts used on the GPU, and the archived
                     determinism probes the reproducibility figures rest on
```

**Not in this repository, by decision:**

- **the raw sweep data** (825 JSONL files, ~613 MB) — to be deposited with a DOI at
  submission, which is the right home for citable data;
- **the degraded audio bank** — it derives from SLURP *audio*, which is CC BY-NC 4.0, and
  inherits that NonCommercial term; it is regenerable from the seed in one command;
- **model weights** (~14 GB, re-downloadable) and the **NOISEX-92** noise track, which is
  never redistributed here in any form;
- **copyrighted PDFs** of third-party papers;
- the project's internal working record — design documents, decision log, reading notes and
  operational runbooks — which is kept in the working repository.

## Running it

Two virtual environments, **not interchangeable** (the split exists because
`parakeet-mlx` and `mlx-lm` require Python ≥ 3.10):

| environment | Python | used for |
|---|---|---|
| `venv/` | 3.9 | Whisper, routing, **all analysis**, pytest |
| `venv312/` | 3.12 | Parakeet, `mlx_lm`, librosa |

Main dependencies: `mlx-whisper`, `parakeet-mlx`, `mlx-lm`, `openai` (the routing client
speaks to any OpenAI-compatible endpoint — `mlx_lm.server` locally, vLLM on the GPU),
`jiwer`, `statsmodels`, `pyroomacoustics`, `librosa`, `soundfile`, `metaphone`.
**Run everything from the repository root.**

```bash
# 1. build the degraded audio bank (deterministic, resume-safe)
./venv/bin/python scripts/corrupt_audio.py --seed 42

# 2. transcription (cascade) and routing
./venv312/bin/python scripts/transcribe_parakeet.py ...
./venv/bin/python  scripts/route_transcripts.py --in <transcripts>.jsonl --out <routed>.jsonl
./venv/bin/python  scripts/route_omni.py        --audio-root corrupted/ ...   # omni arm

# 3. scoring — four stages, chained in this order
#    compute_metrics.py → annotate_errors.py → compute_pf.py → categorize_failures.py

# 4. analysis
./venv/bin/python scripts/fit_mixed_effects.py        # H1, H2a, H2b, H3, H4b
./venv/bin/python scripts/analyze_absorption.py       # absorption, four comparators
./venv/bin/python scripts/mediation_analysis.py       # ASR vs router decomposition

# tests
./venv/bin/python -m pytest tests/ -q
```

⚠️ **On the test suite in this repository.** The full suite is **440 tests and passes in the
working repository**. Here, **18 of them do not run**: 13 spawn `./venv/bin/python` as a
subprocess and therefore need the environment above to exist, and the remaining 5 read the
raw sweep files that are deposited separately. The other 419 pass, plus 3 skips. Nothing is
disabled or deleted to make that number look better.

## Data provenance and licensing

- **SLURP (Bastianelli et al., 2020) has a split licence.** Its *textual* part is CC BY 4.0;
  its *audio* part is **CC BY-NC 4.0 (NonCommercial)**. The degraded audio bank therefore
  inherits the NonCommercial term and is not distributed here.
- **Result files** carry transcriptions, predictions and derived metrics and **no audio**, so
  they follow the textual half; CC BY 4.0 is defensible for them, with attribution to
  Bastianelli et al. (2020).
- **NOISEX-92** babble is never redistributed — a download script plus attribution instead.
- **Code: MIT** (`LICENSE`).

## A note on how this was produced

This project was built with extensive use of AI assistants, under the author's direction and
review: the degradation pipeline, the transcription and routing drivers, the scoring, the
statistical models, the figures and the test suite.

**No data was produced, estimated or modified by an AI.** Every row is the recorded output of
a named model on a named audio file; result files are treated as immutable by project rule
(the scripts refuse to write over their own input, and there are tests that prove it), and a
schema validator checks all 86,944 rows against the documented schema.

Because generated text can state a guess in the same confident register as a checked fact, the
project carries an independent verification layer: one script recomputes every numerical claim
in the project documents from the data files, a second checks the documents against each other
for dead paths and stale or contradictory claims, and the test suite pins the metric formulas
themselves — every absorption formula is implemented once, in `scripts/absorption_metrics.py`,
and a test fails if a definition drifts from the one the design states.

## Citation

An archived, citable version with a DOI will be deposited at submission. Please also cite the
underlying dataset: Bastianelli, E., Vanzo, A., Swietojanski, P., & Rieser, V. (2020).
*SLURP: A Spoken Language Understanding Resource Package.* EMNLP 2020.
