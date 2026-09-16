#!/bin/bash
# =====================================================================
# Aspetta che il server vLLM sia pronto — INTERROGANDOLO SULLA VM.
#
#   ./scripts/vm_wait_server.sh <repo-id-atteso> [MINUTI_MAX]
#
# 🛑 PERCHE' NON SI ASPETTA ATTRAVERSO IL TUNNEL (costato 8,12 USD il
#    2026-08-29). Il ciclo di attesa interrogava
#    http://localhost:8000/v1/models sul Mac, cioe' ATTRAVERSO il tunnel
#    SSH. Quando il tunnel e' caduto (exit 255) mentre Qwen3-Omni
#    caricava 26 GB, quella domanda non poteva piu' distinguere «il
#    server sta ancora caricando» da «non c'e' piu' un tunnel»: il ciclo
#    ha aspettato per sempre e l'A100 e' rimasta accesa a non fare
#    niente fino allo spegnimento automatico.
#    Qui ogni giro apre una connessione SSH nuova e fa `curl` IN LOCALE
#    sulla VM. Se l'SSH e' morto, il giro FALLISCE E LO DICE — non puo'
#    scambiare un silenzio per un'attesa. E' la lezione ricorrente di
#    questo progetto: un controllo che non puo' vedere cio' che sorveglia.
#
# Ogni giro rinnova anche il lease della sicura (vm_ssh.sh), quindi la
# macchina resta accesa finche' qualcuno la sta davvero aspettando — e
# non un minuto di piu'.
# =====================================================================
set -u

EXPECTED=${1:?repo id atteso mancante}
MAX_MIN=${2:-40}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))
GIRO=0

echo "attendo $EXPECTED — interrogando la VM in locale, non il tunnel (max ${MAX_MIN} min)"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  GIRO=$(( GIRO + 1 ))
  TS=$(date -u +%H:%M:%SZ)
  OUT=$("$HERE/vm_ssh.sh" --lease 60 \
        'curl -s --max-time 10 http://127.0.0.1:8000/v1/models; echo; echo "---GPU---"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; echo "---LOG---"; tail -2 ~/vllm_server.log' 2>&1)
  RC=$?

  if [ "$RC" -ne 0 ]; then
    echo "[$TS giro $GIRO] 🛑 SSH FALLITO (rc=$RC) — non e' un'attesa, e' un errore:"
    printf '%s\n' "$OUT" | tail -3 | sed 's/^/    /'
  elif printf '%s' "$OUT" | grep -q "$EXPECTED"; then
    echo "[$TS giro $GIRO] ✅ PRONTO — il server annuncia $EXPECTED"
    printf '%s\n' "$OUT" | sed -n '/---GPU---/,$p' | sed 's/^/    /'
    exit 0
  else
    GPU=$(printf '%s' "$OUT" | sed -n '/---GPU---/{n;p;}')
    LOG=$(printf '%s' "$OUT" | sed -n '/---LOG---/,$p' | tail -1 | cut -c1-120)
    echo "[$TS giro $GIRO] carica… gpu=${GPU:-?} | $LOG"
  fi
  sleep 30
done

echo "🛑 scaduti ${MAX_MIN} minuti senza che il server rispondesse. NON e' pronto."
exit 1
