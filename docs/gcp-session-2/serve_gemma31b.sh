#!/bin/bash
# ============================================================
# Sessione GCP 2 — cascade router Google (H2b/H4)
# vLLM 0.27.1 | torch 2.13.0+cu130 | driver 610.57.04 | A100 cc 8.0
#
# --max-model-len 4096
#   Default 65536. Fabbisogno misurato: prompt ~444 tok + trascrizione
#   piu lunga del dataset ~528 tok + 256 output = ~1227 token. 3.3x margine.
#
# --enforce-eager
#   Senza, vLLM chiama profile_cudagraph_memory() -> OOM misurato due volte
#   il 2026-08-17 (784 MiB richiesti, 419 MiB liberi; pesi 30.2 GB su 39.5).
#   E il modello piu stretto dei quattro (tracker §15). I grafi CUDA sono
#   solo una ottimizzazione di esecuzione: disattivarli non cambia l output.
# ============================================================
export VLLM_USE_FLASHINFER_SAMPLER=0
exec $HOME/vllmenv/bin/vllm serve google/gemma-4-31B-it-qat-w4a16-ct \
  --port 8000 \
  --served-model-name google/gemma-4-31B-it-qat-w4a16-ct \
  --max-model-len 4096 \
  --enforce-eager
