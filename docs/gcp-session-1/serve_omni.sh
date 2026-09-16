#!/bin/bash
# ============================================================
# Sessione GCP 1 — configurazione di serving CONGELATA
# Deve essere IDENTICA per tutti e quattro i modelli della tesi,
# altrimenti una differenza fra modelli puo essere colpa dei flag.
#
# Ambiente misurato sulla macchina 2026-08-17:
#   vLLM 0.27.1 | torch 2.13.0+cu130 | driver NVIDIA 610.57.04
#   A100-SXM4-40GB, compute capability 8.0 (Ampere), 39.5 GB
#
# VLLM_USE_FLASHINFER_SAMPLER=0
#   FlashInfer compila kernel CUDA a runtime e richiede nvcc, che non
#   e installato (abbiamo il driver, non il toolkit). A temperature=0
#   il decoding e greedy (argmax), quindi il campionatore top-k/top-p
#   non viene mai esercitato: si usa l implementazione di riferimento
#   di PyTorch. Nessun effetto atteso sull output; piu riproducibile.
# ============================================================
export VLLM_USE_FLASHINFER_SAMPLER=0
exec $HOME/vllmenv/bin/vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit \
  --port 8000 \
  --served-model-name cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
