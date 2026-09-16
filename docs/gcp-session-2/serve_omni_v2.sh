#!/bin/bash
# ============================================================
# Sessione GCP 4 — Qwen3-Omni: PREPARATO, MAI ESEGUITO.
#
# 🛑 Questo titolo diceva "Qwen3-Omni ri-servito su vLLM 0.28.0" come fatto
#    compiuto. NON e' mai successo. Il comando concatenato che doveva fermare il
#    27B e avviare questo server usava `pkill -f "$P1$P2"` dentro
#    `gcloud compute ssh --command=`; la shell LOCALE ha espanso le variabili
#    prima di spedire, quindi la riga di comando remota conteneva il letterale
#    `vllmenv2/bin/vllm serve` e `pkill` ha ucciso la propria sessione SSH
#    (uscita 255 — docs/session-2026-08-27-logs/finish.log, 2026-08-27 16:01 UTC).
#    Il server non e' mai partito, la catena ha atteso 50 minuti e ha spento la
#    macchina. Conseguenza: il braccio `omni_qwen` nei risultati e' ancora dato
#    vLLM 0.27.1, mentre gli altri tre bracci serviti da vLLM sono 0.28.0.
#    Lo scarto va DICHIARATO in Methods; il limite di quanto puo' costare e' la
#    misura dell'effetto-versione in §13. Vedi anche §14 (parte 4 non raggiunta).
#    Questo script resta qui come cio' che sarebbe stato eseguito, non come record.
#
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
exec $HOME/vllmenv2/bin/vllm serve cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit \
  --port 8000 \
  --served-model-name cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
