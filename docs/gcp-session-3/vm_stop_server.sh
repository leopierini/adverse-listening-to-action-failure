#!/bin/bash
# Ferma il server vLLM usando il PID registrato — mai un pattern.
# Da eseguire SULLA VM.
set -u
PIDFILE="$HOME/vllm_server.pid"
[ -f "$PIDFILE" ] || { echo "nessun pidfile: niente da fermare"; exit 0; }
PID=$(cat "$PIDFILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
  echo "fermato pid $PID"
else
  echo "pid $PID gia' morto"
fi
rm -f "$PIDFILE"
# La VRAM deve tornare libera prima di servire il modello successivo.
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
