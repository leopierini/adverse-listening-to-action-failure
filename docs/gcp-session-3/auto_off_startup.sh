#!/bin/bash
# =====================================================================
# DEAD-MAN'S SWITCH — installato come `startup-script` nei metadata
# dell'istanza, quindi eseguito da GCE a OGNI avvio, come root,
# prima e indipendentemente da qualunque sessione SSH.
#
# PERCHE' QUI E NON NEL TERMINALE DI CLAUDE (richiesta di Leonardo,
# 2026-08-28): la macchina si deve spegnere anche se la sessione di
# Claude muore, se finisce il consumo settimanale dell'abbonamento, se
# cade l'SSH, o se GCE la riaccende da sola (questa istanza ha
# automaticRestart=True / onHostMaintenance=TERMINATE — letti con
# `gcloud compute instances describe` il 2026-08-28).
#
# L'ERRORE DEL 2026-08-28 CHE QUESTO EVITA (§12): la vecchia sicura
# chiedeva "il heartbeat e' vecchio?" invece di "la macchina e' accesa e
# incustodita?". Mentre la VM era spenta il heartbeat invecchiava da
# solo, quindi era GARANTITO che scattasse al primo avvio dopo una
# pausa — e infatti spense una VM appena avviata dopo 79 secondi.
# Qui il conto alla rovescia parte dalla transizione a RUNNING per
# costruzione: lo script esiste solo perche' la macchina si e' avviata.
# Non c'e' nessuno stato che possa invecchiare mentre e' spenta.
# =====================================================================
set -u

DEADLINE_MIN=150
MARK=/var/run/auto_off.armed

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

# --- Traccia verificabile: cosa e' stato armato, e quando -----------
{
  echo "armed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "deadline_min=${DEADLINE_MIN}"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
} > "${MARK}" 2>/dev/null

exit 0
