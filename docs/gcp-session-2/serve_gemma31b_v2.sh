#!/bin/bash
# ============================================================
# Sessione GCP 4 — Gemma su vLLM 0.28.0 (ambiente vllmenv2)
# Ambiente: $HOME/vllmenv2 (vedi la riga `exec` in fondo) — vLLM 0.28.0 con
#   transformers 5.16.1, letti SULLA VM durante la sessione del 2026-08-27.
#   ⚠️ NON riverificabili da questo repo: la VM e' TERMINATED. I controlli che li
#   risolvono al prossimo avvio sono `$HOME/vllmenv2/bin/vllm --version` e
#   `$HOME/vllmenv2/bin/pip show transformers`.
#   ⚠️ Le note di rilascio di 0.28.0 dicevano transformers 5.15.0; 5.16.1 e' cio'
#   che pip ha effettivamente risolto. La correzione e' in §14.
#   ⚠️ La versione di torch dentro vllmenv2 NON e' stata misurata. Questo header
#   diceva "vLLM 0.27.1 | torch 2.13.0+cu130": era l'ambiente vllmenv, e sotto
#   questo script non vale — vllmenv (0.27.1) resta intatto e separato.
# Driver NVIDIA 610.57.04 | A100-SXM4-40GB, compute capability 8.0 (Ampere), 39.5 GB
#   — invariati, misurati 2026-08-17 (§15): il secondo ambiente Python non li tocca.
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
exec $HOME/vllmenv2/bin/vllm serve google/gemma-4-31B-it-qat-w4a16-ct \
  --port 8000 \
  --served-model-name google/gemma-4-31B-it-qat-w4a16-ct \
  --max-model-len 4096 \
  --enforce-eager
