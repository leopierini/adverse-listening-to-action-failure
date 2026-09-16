#!/bin/bash
# ============================================================
# Sessione GCP 4 — omni Google, stessi flag, ambiente vllmenv2 (vLLM 0.28.0)
# Deve essere IDENTICA per tutti e quattro i modelli della tesi,
# altrimenti una differenza fra modelli puo essere colpa dei flag.
#
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
# VLLM_USE_FLASHINFER_SAMPLER=0
#   FlashInfer compila kernel CUDA a runtime e richiede nvcc, che non
#   e installato (abbiamo il driver, non il toolkit). A temperature=0
#   il decoding e greedy (argmax), quindi il campionatore top-k/top-p
#   non viene mai esercitato: si usa l implementazione di riferimento
#   di PyTorch. Nessun effetto atteso sull output; piu riproducibile.
# ============================================================
export VLLM_USE_FLASHINFER_SAMPLER=0
exec $HOME/vllmenv2/bin/vllm serve google/gemma-4-12B-it-qat-w4a16-ct \
  --port 8000 \
  --served-model-name google/gemma-4-12B-it-qat-w4a16-ct
