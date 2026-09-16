#!/bin/bash
# =====================================================================
# Verifica che la sicura sia DAVVERO armata su QUESTO avvio.
# Da eseguire SULLA VM, subito dopo il boot.
#
# Non chiede "esiste lo script?" ma "questo boot ha uno spegnimento
# programmato?" — la distinzione che il 2026-08-28 e' costata una VM
# spenta dopo 79 secondi (§12): la vecchia sicura testava una
# condizione diversa da quella che intendeva dire.
# Uscita 0 = armata. Uscita 1 = NON armata, spegni a mano e indaga.
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

if [ "$RC" -eq 0 ]; then
  echo "RISULTATO: sicura ARMATA su questo boot (entrambi i livelli)."
else
  echo "RISULTATO: sicura INCOMPLETA — non lasciare la macchina incustodita."
fi
exit "$RC"
