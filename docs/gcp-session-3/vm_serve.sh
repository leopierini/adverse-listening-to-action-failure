#!/bin/bash
# =====================================================================
# Avvia UNO dei serve script congelati e ne REGISTRA IL PID.
# Da eseguire SULLA VM.  Uso:  ./vm_serve.sh serve_gemma12b_v2.sh
#
# 🛑 PERCHE' IL PID E NON `pkill -f` (§12 — e' costato due volte in un
#    solo giorno, 2026-08-27 alle 01:31 e alle 16:01 UTC).
#    `pkill -f` confronta le RIGHE DI COMANDO INTERE, e la shell remota
#    che esegue il tuo comando CONTIENE quella stringa: quindi `pkill`
#    uccide la propria sessione SSH (uscita 255). La "correzione"
#    scritta allora — spezzare il pattern in due variabili — fallisce
#    lo stesso, perche' la shell LOCALE espande le variabili prima di
#    spedire il comando. La seconda volta e' costata il re-serve di
#    Qwen3-Omni, 50 minuti di A100 inattiva (~3,1 USD) e l'uniformita'
#    di versione fra i bracci.
#    Un PID non e' un pattern: non puo' corrispondere a se stesso.
# =====================================================================
set -u
SCRIPT_NAME="${1:?serve script mancante}"
PIDFILE="$HOME/vllm_server.pid"
LOG="$HOME/vllm_server.log"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "🛑 un server e' gia' vivo con pid $(cat "$PIDFILE") — fermalo prima (vm_stop_server.sh)"
  exit 1
fi

: > "$LOG"
nohup bash "$HOME/$SCRIPT_NAME" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "avviato $SCRIPT_NAME con pid $(cat "$PIDFILE"); log -> $LOG"
