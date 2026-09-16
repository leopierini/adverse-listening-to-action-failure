#!/bin/bash
# =====================================================================
# Verifica che la sicura sia DAVVERO armata su QUESTO avvio — v2.
# Da eseguire SULLA VM, subito dopo il boot.
#
# Non chiede "esiste lo script?" ma "questo boot ha uno spegnimento
# programmato?" — la distinzione che il 2026-08-28 e' costata una VM
# spenta dopo 79 secondi (§12): la vecchia sicura testava una
# condizione diversa da quella che intendeva dire.
#
# v2 aggiunge il livello C (spegnimento per inattivita'): non basta che
# il servizio esista, deve essere ATTIVO e deve aver trovato gli
# strumenti con cui misura. Un guardiano che non trova nvidia-smi non
# sta guardando niente.
# Uscita 0 = armata su tutti i livelli. Uscita 1 = incompleta.
# =====================================================================
set -u
RC=0

echo "--- boot corrente ---"
echo "boot_id  : $(cat /proc/sys/kernel/random/boot_id)"
echo "uptime_s : $(cut -d. -f1 /proc/uptime)"
echo "adesso   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

echo "--- traccia lasciata dallo startup-script ---"
if [ -f /var/run/auto_off.armed ]; then
  cat /var/run/auto_off.armed
  grep -q '^switch_version=2$' /var/run/auto_off.armed || {
    echo "⚠️ il marker NON dice switch_version=2 — sta girando una sicura vecchia"; RC=1; }
else
  echo "MARKER ASSENTE — lo startup-script non ha girato su questo boot"
  RC=1
fi
echo

echo "--- livello A: spegnimento programmato (systemd) ---"
if [ -f /run/systemd/shutdown/scheduled ]; then
  cat /run/systemd/shutdown/scheduled
  USEC=$(sed -n 's/^USEC=//p' /run/systemd/shutdown/scheduled)
  if [ -n "${USEC:-}" ]; then
    echo "spegnimento previsto : $(date -u -d "@$((USEC/1000000))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)"
    echo "fra                  : $(( (USEC/1000000 - $(date +%s)) / 60 )) minuti"
  fi
else
  echo "NESSUNO spegnimento programmato — livello A non armato"
  RC=1
fi
echo

echo "--- livello B: timer transitorio di backstop ---"
if systemctl list-timers --all 2>/dev/null | grep -qi auto-off; then
  systemctl list-timers --all 2>/dev/null | grep -i auto-off
else
  echo "timer di backstop assente"
  RC=1
fi
echo

echo "--- livello C: spegnimento per inattivita' ---"
if systemctl is-active --quiet auto-off-idle 2>/dev/null; then
  echo "servizio auto-off-idle : ATTIVO (pid $(systemctl show -p MainPID --value auto-off-idle 2>/dev/null))"
else
  echo "servizio auto-off-idle : NON attivo"
  RC=1
fi
if [ -f /var/log/auto_off_idle.log ]; then
  echo "strumenti che ha trovato:"
  grep -m1 '^\[.*strumenti' /var/log/auto_off_idle.log | sed 's/^/  /'
  grep -q 'ASSENTE' <(grep -m1 'strumenti' /var/log/auto_off_idle.log) && {
    echo "  ⚠️ un segnale su due non e' misurabile su questo boot"; RC=1; }
  echo "ultime osservazioni:"
  tail -3 /var/log/auto_off_idle.log | sed 's/^/  /'
else
  echo "nessun log del livello C — non ha mai girato"
  RC=1
fi
if [ -f /var/run/vm_lease_until ]; then
  L=$(tr -dc '0-9' < /var/run/vm_lease_until)
  echo "lease    : fino a $(date -u -d "@$L" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null) (fra $(( (L - $(date +%s)) / 60 )) minuti)"
  echo "           rinnovalo con:  /usr/local/sbin/vm_lease.sh 45"
else
  echo "lease    : ASSENTE — la macchina si spegnera' appena la GPU e' ferma da 20 minuti"
fi
echo

if [ "$RC" -eq 0 ]; then
  echo "RISULTATO: sicura ARMATA su questo boot (livelli A, B e C)."
else
  echo "RISULTATO: sicura INCOMPLETA — non lasciare la macchina incustodita."
fi
exit "$RC"
