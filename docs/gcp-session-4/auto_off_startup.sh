#!/bin/bash
# =====================================================================
# DEAD-MAN'S SWITCH v2 — installato come `startup-script` nei metadata
# dell'istanza, quindi eseguito da GCE a OGNI avvio, come root, prima e
# indipendentemente da qualunque sessione SSH.
#
# PERCHE' QUI E NON NEL TERMINALE DI CLAUDE (richiesta di Leonardo,
# 2026-08-28, ribadita 2026-08-30): la macchina si deve spegnere anche
# se la sessione di Claude muore, se finisce il consumo settimanale
# dell'abbonamento, se cade l'SSH, o se GCE la riaccende da sola
# (questa istanza ha automaticRestart=True / onHostMaintenance=TERMINATE
# — letti con `gcloud compute instances describe` il 2026-08-28).
#
# L'ERRORE DEL 2026-08-28 CHE QUESTO EVITA (§12): la vecchia sicura
# chiedeva "il heartbeat e' vecchio?" invece di "la macchina e' accesa e
# incustodita?". Mentre la VM era spenta il heartbeat invecchiava da
# solo, quindi era GARANTITO che scattasse al primo avvio dopo una
# pausa — e infatti spense una VM appena avviata dopo 79 secondi.
# Qui il conto alla rovescia parte dalla transizione a RUNNING per
# costruzione: lo script esiste solo perche' la macchina si e' avviata.
# Non c'e' nessuno stato che possa invecchiare mentre e' spenta.
#
# ---------------------------------------------------------------------
# COSA CAMBIA IN v2 (2026-08-30) — il livello C.
#
# v1 aveva un solo esito: 150 minuti dall'avvio, comunque siano andate
# le cose. Il 2026-08-29 questo ha significato pagare 8,12 USD di A100
# ferma dopo che il tunnel era caduto: la sicura ha funzionato, ma il
# suo unico verdetto arriva DUE ORE dopo che la macchina ha smesso di
# servire a qualcosa.
#
# Il livello C spegne quando NESSUNO STA USANDO LA GPU. Misura due
# segnali indipendenti e li considera "attivita'":
#   1. utilizzo GPU da `nvidia-smi` >= UTIL_BUSY_PCT in un campione;
#   2. il contatore delle richieste completate di vLLM che avanza
#      (letto su 127.0.0.1, MAI attraverso il tunnel).
# e un terzo segnale che tiene la macchina VIVA anche se e' inattiva:
#   3. il LEASE, /var/run/vm_lease_until, un istante UTC futuro scritto
#      da chi sta presidiando la macchina (Claude, prima di ogni fase
#      lunga: caricamento pesi, sweep). Assente o scaduto = nessuno
#      la sta presidiando.
#
# La polarita' e' quella giusta: per restare accesa serve una prova
# positiva (attivita' o lease valido). L'assenza di prove spegne.
# Un lease e' un istante nel FUTURO: non puo' invecchiare "a favore"
# mentre la macchina e' spenta, che e' l'errore di v1 del heartbeat.
# =====================================================================
set -u

DEADLINE_MIN=150          # livello A — tetto rigido, invariato da v1
IDLE_MIN=20               # livello C — minuti di GPU ferma prima di spegnere
INITIAL_LEASE_MIN=35      # livello C — presidenza concessa d'ufficio all'avvio
UTIL_BUSY_PCT=5           # livello C — sopra questo utilizzo la GPU "lavora"
SAMPLE_SEC=20             # livello C — ogni quanto si campiona
MARK=/var/run/auto_off.armed
LEASE=/var/run/vm_lease_until
IDLE_SCRIPT=/usr/local/sbin/auto_off_idle.sh
LEASE_SCRIPT=/usr/local/sbin/vm_lease.sh

# --- Livello A: spegnimento schedulato a livello di sistema ---------
/sbin/shutdown -h +${DEADLINE_MIN} \
  "auto-off: tetto rigido di ${DEADLINE_MIN} minuti dall'avvio" </dev/null >/dev/null 2>&1

# --- Livello B: timer transitorio systemd, indipendente dal livello A -
# Se `shutdown` non fosse armato (init diverso, logind assente), questo
# resta. Ridondante di proposito: e' la sicura, non una comodita'.
/usr/bin/systemd-run \
  --unit=auto-off-backstop \
  --on-active=$((DEADLINE_MIN + 10))min \
  --timer-property=AccuracySec=10s \
  /sbin/shutdown -h now </dev/null >/dev/null 2>&1

# --- Livello C: il lease, scrivibile senza sudo ----------------------
# 666 di proposito: chi lavora sulla macchina deve poter rinnovare la
# presidenza dentro lo stesso comando SSH che sta gia' eseguendo, senza
# una seconda autenticazione che possa fallire in silenzio.
echo $(( $(date +%s) + INITIAL_LEASE_MIN * 60 )) > "${LEASE}" 2>/dev/null
chmod 666 "${LEASE}" 2>/dev/null

cat > "${LEASE_SCRIPT}" <<'LEASE_EOF'
#!/bin/bash
# Estende la presidenza della macchina di N minuti (default 30).
#   ./vm_lease.sh 45
# Scrive un istante UTC futuro in /var/run/vm_lease_until. Finche' quell
# istante e' nel futuro il livello C non spegne, anche a GPU ferma.
set -u
MIN="${1:-30}"
LEASE=/var/run/vm_lease_until
UNTIL=$(( $(date +%s) + MIN * 60 ))
echo "$UNTIL" > "$LEASE"
echo "lease fino a $(date -u -d "@$UNTIL" +%Y-%m-%dT%H:%M:%SZ) (+${MIN} min)"
LEASE_EOF
chmod 755 "${LEASE_SCRIPT}" 2>/dev/null

cat > "${IDLE_SCRIPT}" <<IDLE_EOF
#!/bin/bash
# Livello C — spegne quando nessuno sta usando la GPU.
# Scritto dallo startup-script a ogni avvio: la copia su disco non puo'
# divergere da quella nei metadata.
set -u
# systemd non eredita il PATH di una shell di login: senza questa riga
# \`nvidia-smi\` e \`curl\` potrebbero non essere trovati, il segnale 1 e il
# segnale 2 tacerebbero entrambi, e la sicura si ridurrebbe al lease.
# Un controllo che non trova lo strumento con cui misura e' esattamente
# il modo in cui questo progetto si e' gia' fatto male una volta.
export PATH="\${PATH:+\$PATH:}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
IDLE_MIN=${IDLE_MIN}
UTIL_BUSY_PCT=${UTIL_BUSY_PCT}
SAMPLE_SEC=${SAMPLE_SEC}
LEASE=${LEASE}
LOG=/var/log/auto_off_idle.log

log() { echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] \$*" >> "\$LOG"; }

last_activity=\$(date +%s)
prev_reqs=""
log "livello C avviato | idle_min=\${IDLE_MIN} util_busy_pct=\${UTIL_BUSY_PCT} sample_sec=\${SAMPLE_SEC}"
# Si scrive QUALE binario e' stato trovato, non che "dovrebbe esserci":
# se una di queste due righe dice 'ASSENTE', quel segnale non esiste su
# questo avvio e va saputo leggendo il log, non indovinato.
log "strumenti | nvidia-smi=\$(command -v nvidia-smi || echo ASSENTE) curl=\$(command -v curl || echo ASSENTE)"

while true; do
  now=\$(date +%s)
  busy=0
  why=""

  # segnale 1 — la GPU sta eseguendo kernel
  util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  if [ -n "\$util" ] && [ "\$util" -ge "\$UTIL_BUSY_PCT" ] 2>/dev/null; then
    busy=1; why="gpu_util=\${util}%"
  fi

  # segnale 2 — vLLM ha completato altre richieste (letto in LOCALE,
  # mai attraverso il tunnel: e' il tunnel a poter mentire)
  reqs=\$(curl -s --max-time 3 http://127.0.0.1:8000/metrics 2>/dev/null \\
          | awk '/^vllm:request_success_total/ {s=s "|" \$NF} END {print s}')
  if [ -n "\$reqs" ] && [ -n "\$prev_reqs" ] && [ "\$reqs" != "\$prev_reqs" ]; then
    busy=1; why="\${why} vllm_requests_avanzate"
  fi
  [ -n "\$reqs" ] && prev_reqs="\$reqs"

  [ "\$busy" -eq 1 ] && last_activity=\$now

  # segnale 3 — qualcuno dichiara di stare presidiando la macchina
  lease=\$(tr -dc '0-9' < "\$LEASE" 2>/dev/null)
  [ -z "\$lease" ] && lease=0

  idle_s=\$(( now - last_activity ))
  if [ "\$now" -ge "\$lease" ] && [ "\$idle_s" -ge \$(( IDLE_MIN * 60 )) ]; then
    log "SPENGO: gpu ferma da \$(( idle_s / 60 )) min (soglia \${IDLE_MIN}) e lease scaduto da \$(( now - lease )) s"
    /sbin/shutdown -h now "auto-off livello C: nessuno sta usando la GPU" </dev/null >/dev/null 2>&1
    exit 0
  fi

  # una riga ogni 5 minuti, per poter ricostruire cosa ha visto
  if [ \$(( now % 300 )) -lt "\$SAMPLE_SEC" ]; then
    log "vivo | busy=\${busy} \${why} | ferma da \$(( idle_s / 60 )) min | lease fra \$(( (lease - now) / 60 )) min"
  fi

  sleep "\$SAMPLE_SEC"
done
IDLE_EOF
chmod 755 "${IDLE_SCRIPT}" 2>/dev/null

/usr/bin/systemd-run \
  --unit=auto-off-idle \
  --collect \
  --description="auto-off livello C: spegne la VM quando nessuno usa la GPU" \
  "${IDLE_SCRIPT}" </dev/null >/dev/null 2>&1

# --- Traccia verificabile: cosa e' stato armato, e quando -----------
{
  echo "armed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "deadline_min=${DEADLINE_MIN}"
  echo "idle_min=${IDLE_MIN}"
  echo "initial_lease_min=${INITIAL_LEASE_MIN}"
  echo "switch_version=2"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
} > "${MARK}" 2>/dev/null

exit 0
