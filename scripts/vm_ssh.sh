#!/bin/bash
# =====================================================================
# Un comando sulla VM, con il lease rinnovato nello stesso viaggio.
#
#   ./scripts/vm_ssh.sh 'nvidia-smi'
#   ./scripts/vm_ssh.sh --lease 90 'bash ~/vm_serve.sh serve_omni_v2.sh'
#   ./scripts/vm_ssh.sh --put file1 file2         (copia sulla VM, in ~)
#
# 🛑 PERCHE' IL LEASE VIENE RINNOVATO QUI E NON DA UN PROCESSO A PARTE.
#    Il livello C della sicura spegne la macchina quando nessuno la
#    presidia. "Presidiare" non e' una promessa fatta una volta: e' il
#    fatto che qualcuno le stia ancora parlando. Rinnovare il lease
#    dentro lo stesso comando SSH che si sta gia' eseguendo lega le due
#    cose per costruzione — se questa sessione muore, nessuno rinnova, e
#    la macchina si spegne da sola. Un rinnovatore separato che
#    sopravvivesse alla sessione disarmerebbe la sicura proprio nel caso
#    per cui esiste.
#
# 🛑 PERCHE' `ssh` E NON `gcloud compute ssh`: quest'ultimo, con
#    --ssh-flag="-L …", ha lasciato un processo vivo con NIENTE in
#    ascolto sulla 8000 (2026-08-17, runbook §3).
#
# 🛑 GLI ERRORI NON SI NASCONDONO. Il 2026-08-30 un ciclo di attesa con
#    `2>/dev/null` ha riprovato otto volte senza dire perche': la
#    macchina stava semplicemente ancora avviandosi, ma dal log non si
#    poteva sapere. Qui stderr passa.
# =====================================================================
set -u

INST=${THESIS_VM_INSTANCE:-instance-pierini-gpu}
ZONE=${THESIS_VM_ZONE:-europe-west4-a}
USER_ON_VM=${THESIS_VM_USER:-leonardo}
KEY=${THESIS_VM_KEY:-$HOME/.ssh/google_compute_engine}
LEASE_MIN=45

if [ "${1:-}" = "--lease" ]; then LEASE_MIN="$2"; shift 2; fi

IP=${THESIS_VM_IP:-$(gcloud compute instances describe "$INST" --zone="$ZONE" \
      --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null)}
if [ -z "$IP" ]; then
  echo "🛑 nessun IP: la macchina e' spenta o gcloud non risponde" >&2
  exit 3
fi

SSH_ARGS=(-i "$KEY"
          -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null
          -o LogLevel=ERROR
          -o ConnectTimeout=20
          -o ServerAliveInterval=30
          -o ServerAliveCountMax=3)

if [ "${1:-}" = "--put" ]; then
  shift
  scp "${SSH_ARGS[@]}" "$@" "${USER_ON_VM}@${IP}:~/"
  exit $?
fi

REMOTE="/usr/local/sbin/vm_lease.sh ${LEASE_MIN} >/dev/null 2>&1; $*"
exec ssh "${SSH_ARGS[@]}" "${USER_ON_VM}@${IP}" "$REMOTE"
