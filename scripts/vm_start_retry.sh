#!/bin/bash
# =====================================================================
# UN TENTATIVO di accensione della VM. Non e' un loop: e' un tick.
# Lo chiama launchd ogni INTERVAL secondi (vedi scripts/vm_retry_ctl.sh),
# percio' sopravvive alla chiusura del terminale, alla morte della
# sessione di Claude e alla fine dei crediti dell'abbonamento — che e'
# esattamente la richiesta di Leonardo del 2026-08-30.
#
# Il loop precedente (docs/gcp-session-3/start_retry_loop.sh) era un
# `while` dentro la sessione: moriva col terminale. Le tre regole che
# aveva imparato a caro prezzo restano valide e sono qui sotto.
#
# 🛑 REGOLA 1 (misurata 2026-08-20): una risposta STOCKOUT NON significa
#    che la macchina sia rimasta spenta. Quel giorno `start` rispose
#    STOCKOUT e l'istanza si avvio' LO STESSO, fatturando. Non si legge
#    MAI l'esito di `start`: si legge sempre lo STATO.
#
# 🛑 REGOLA 2 (misurata 2026-08-28): `gcloud instances start` senza
#    --async resta appeso a fare polling, e quel tempo si SOMMA
#    all'intervallo: 28 minuti medi invece di 10, con punte di 48.
#
# 🛑 REGOLA 3: se lo stato e' gia' transitorio (STAGING/PROVISIONING/
#    STOPPING/SUSPENDING) NON si invia un altro start, o si accodano
#    operazioni sulla stessa istanza.
#
# 🛑 REGOLA 4 (nuova, 2026-08-30): il tick si spegne DA SOLO. Due
#    condizioni indipendenti scrivono il file `stop`, e con `stop`
#    presente il tick non chiama nemmeno gcloud:
#      - la macchina e' stata vista RUNNING (lo scopo e' raggiunto);
#      - la scadenza e' passata (default 24 h dall'armamento).
#    Senza questo, un loop che sopravvive alla sessione riaccenderebbe
#    la macchina all'infinito, 10 minuti dopo ogni auto-spegnimento.
#    La sicura di bordo (livello C, docs/gcp-session-4/auto_off_startup.sh)
#    tiene la macchina accesa almeno ~35 minuti se nessuno la presidia,
#    e 35 > INTERVAL: e' garantito che almeno un tick la veda RUNNING.
# =====================================================================
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INST=${THESIS_VM_INSTANCE:-instance-pierini-gpu}
ZONE=${THESIS_VM_ZONE:-europe-west4-a}
GCLOUD=${THESIS_GCLOUD_BIN:-/opt/homebrew/bin/gcloud}
STATE=${THESIS_VM_RETRY_STATE:-$REPO/logs/vm_retry}
LOG="$STATE/retry.log"

mkdir -p "$STATE"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
log() { echo "[$TS] $*" >> "$LOG"; }

# --- REGOLA 4a: gia' fermo -------------------------------------------
if [ -f "$STATE/stop" ]; then
  exit 0
fi

# --- REGOLA 4b: scadenza ---------------------------------------------
DEADLINE=$(cat "$STATE/deadline" 2>/dev/null | tr -dc '0-9')
if [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
  log "🛑 scadenza raggiunta — smetto di riprovare. La macchina NON e' accesa."
  echo "scadenza" > "$STATE/stop"
  exit 0
fi

# --- REGOLA 1: si legge lo STATO, mai l'esito di start ---------------
STATUS=$("$GCLOUD" compute instances describe "$INST" --zone="$ZONE" \
           --format="value(status)" 2>/dev/null | tr -d '[:space:]')

if [ -z "$STATUS" ]; then
  log "stato ILLEGGIBILE (gcloud non ha risposto) — riprovo al tick successivo"
  exit 0
fi

if [ "$STATUS" = "RUNNING" ]; then
  log "✅ RUNNING — scopo raggiunto, smetto. La sicura si e' armata da sola all'avvio."
  echo "running" > "$STATE/stop"
  exit 0
fi

# --- REGOLA 3: niente start sopra un'operazione in corso -------------
case "$STATUS" in
  STAGING|PROVISIONING|STOPPING|SUSPENDING|REPAIRING)
    log "stato transitorio ($STATUS) — non invio un altro start"
    exit 0
    ;;
esac

# Per imparare, senza agire: STATUS.md documenta che un start FALLITO
# muove lastStopTimestamp. Se muova anche lastStartTimestamp non e' mai
# stato misurato, quindi qui si REGISTRA e basta — usarlo come segnale
# di "e' partita" prima di averlo misurato spegnerebbe il loop al primo
# STOCKOUT.
STAMPS=$("$GCLOUD" compute instances describe "$INST" --zone="$ZONE" \
           --format="value(lastStartTimestamp,lastStopTimestamp)" 2>/dev/null | tr '\t' ' ')

ATTEMPT=$(( $(cat "$STATE/attempts" 2>/dev/null | tr -dc '0-9' || echo 0) + 1 ))
echo "$ATTEMPT" > "$STATE/attempts"

# --- REGOLA 2: --async, la richiesta si invia e si torna subito ------
OUT=$("$GCLOUD" compute instances start "$INST" --zone="$ZONE" --async 2>&1)
OP=$(printf '%s' "$OUT" | grep -o 'operation-[0-9a-z-]*' | head -1)
log "tentativo $ATTEMPT | stato_prima=$STATUS | op=${OP:-nessuna} | stamps=${STAMPS:-illeggibili} | $(printf '%s' "$OUT" | tr '\n' ' ' | cut -c1-110)"
exit 0
