#!/bin/bash
# =====================================================================
# Tunnel SSH verso il server vLLM, che si RIAPRE da solo.
#
#   ./scripts/vm_tunnel.sh            (resta in primo piano; Ctrl-C per chiudere)
#
# 🛑 Il 2026-08-29 il tunnel e' caduto (exit 255) mentre Qwen3-Omni
#    caricava, e nulla lo ha riaperto. Il tunnel e' l'unico pezzo di
#    questa catena che sta fra due macchine, quindi e' l'unico che cade
#    per conto suo: va sorvegliato separatamente da cio' che ci passa
#    dentro. Qui ogni caduta viene detta e la connessione riaperta.
#
# 🛑 `ssh` semplice, MAI `gcloud compute ssh --ssh-flag="-L …"`: il
#    2026-08-17 quest'ultimo ha lasciato un processo vivo con NIENTE in
#    ascolto sulla 8000.
#
# Il tunnel muore con questa sessione, di proposito: se la sessione
# finisce nessuno rinnova il lease e la sicura (livello C) spegne la VM.
# =====================================================================
set -u

INST=${THESIS_VM_INSTANCE:-instance-pierini-gpu}
ZONE=${THESIS_VM_ZONE:-europe-west4-a}
USER_ON_VM=${THESIS_VM_USER:-leonardo}
KEY=${THESIS_VM_KEY:-$HOME/.ssh/google_compute_engine}
PORT=${THESIS_VM_PORT:-8000}

IP=$(gcloud compute instances describe "$INST" --zone="$ZONE" \
       --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)
[ -n "$IP" ] || { echo "🛑 nessun IP: la macchina e' spenta" >&2; exit 3; }

APERTURE=0
while true; do
  APERTURE=$(( APERTURE + 1 ))
  echo "[$(date -u +%H:%M:%SZ)] apertura $APERTURE del tunnel ${PORT} -> ${IP}:${PORT}"
  ssh -i "$KEY" -N -L "${PORT}:localhost:${PORT}" \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      "${USER_ON_VM}@${IP}"
  RC=$?
  echo "[$(date -u +%H:%M:%SZ)] 🛑 il tunnel e' caduto (rc=$RC) — riapro fra 3 s"
  sleep 3
done
