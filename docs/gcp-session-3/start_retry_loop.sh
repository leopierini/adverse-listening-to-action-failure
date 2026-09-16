#!/bin/bash
# =====================================================================
# Loop di riavvio per lo slot A100. I tentativi falliti NON sono
# fatturati (§12): il ciclo costa solo tempo.
#
# 🛑 REGOLA 1 (misurata 2026-08-20, runbook §1): una risposta STOCKOUT
#    NON significa che la macchina sia rimasta spenta. Quel giorno
#    `start` rispose STOCKOUT e l'istanza si avvio' LO STESSO,
#    fatturando. Quindi non si legge MAI l'esito di `start`: si legge
#    sempre lo STATO, ed e' quello a decidere.
#
# 🛑 REGOLA 2 (misurata 2026-08-28, questa sessione): la versione
#    sincrona di questo loop otteneva 28 minuti di media fra i
#    tentativi invece dei 10 voluti, con gap fino a 48 minuti, perche'
#    `gcloud instances start` resta appeso a fare polling prima di
#    rispondere e quel tempo si SOMMA all'attesa. Con --async la
#    richiesta si invia e si torna subito, cosi' l'intervallo e'
#    davvero l'intervallo.
#
# 🛑 REGOLA 3: se lo stato e' gia' transitorio (STAGING/PROVISIONING/
#    STOPPING/SUSPENDING) NON si invia un altro start, o si accodano
#    operazioni sulla stessa istanza. Si aspetta e si rilegge.
# =====================================================================
set -u
INST=instance-pierini-gpu
ZONE=europe-west4-a
INTERVAL_SEC=600          # 10 minuti VERI, ora che start non blocca
MAX_ATTEMPTS=48
LOG="$1"

echo "=== loop (async) avviato $(date -u +%Y-%m-%dT%H:%M:%SZ) UTC | ogni ${INTERVAL_SEC}s | max ${MAX_ATTEMPTS} tentativi ===" >> "$LOG"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  STATUS=$(gcloud compute instances describe "$INST" --zone="$ZONE" \
             --format="value(status)" 2>/dev/null)
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  if [ "$STATUS" = "RUNNING" ]; then
    echo "[$TS] ✅ RUNNING dopo $attempt tentativi — esco. La sicura si e' armata da sola all'avvio." >> "$LOG"
    exit 0
  fi

  case "$STATUS" in
    STAGING|PROVISIONING|STOPPING|SUSPENDING)
      echo "[$TS] stato transitorio ($STATUS) — aspetto senza inviare un altro start" >> "$LOG"
      sleep 20
      continue
      ;;
  esac

  attempt=$((attempt + 1))
  OUT=$(gcloud compute instances start "$INST" --zone="$ZONE" --async 2>&1)
  OP=$(printf '%s' "$OUT" | grep -o 'operation-[0-9a-z-]*' | head -1)
  echo "[$TS] tentativo $attempt | stato_prima=${STATUS:-ILLEGGIBILE} | op=${OP:-nessuna} | $(printf '%s' "$OUT" | tr '\n' ' ' | cut -c1-110)" >> "$LOG"

  sleep "$INTERVAL_SEC"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 🛑 arreso dopo ${MAX_ATTEMPTS} tentativi. La macchina NON e' accesa." >> "$LOG"
exit 1
