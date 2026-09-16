#!/bin/bash
# =====================================================================
# Arma / interroga / disarma il ritentativo persistente di accensione.
#
#   ./scripts/vm_retry_ctl.sh arm [ORE]   (default 24)
#   ./scripts/vm_retry_ctl.sh status
#   ./scripts/vm_retry_ctl.sh disarm
#
# "Persistente" vuol dire: e' launchd a chiamare il tick, non una shell.
# Sopravvive alla chiusura del terminale e alla morte della sessione di
# Claude. Gira nella sessione grafica dell'utente: se il Mac dorme, i
# tick non partono — e va benissimo, e' la direzione sicura.
#
# Il tick si ferma DA SOLO (regola 4 in vm_start_retry.sh): quando vede
# RUNNING, o alla scadenza. `disarm` e' per fermarlo prima.
#
# 🛑 PERCHE' IL TICK GIRA DA ~/.thesis_vm_retry E NON DAL REPO
#    (misurato 2026-08-30). Il repo sta sotto ~/Desktop, che macOS
#    protegge con TCC. Un LaunchAgent che punta la' non riesce nemmeno a
#    leggere lo script: launchd registra `last exit code = 126` e
#    `/bin/bash: .../vm_start_retry.sh: Operation not permitted`, e il
#    loop non parte MAI — silenziosamente, perche' il suo log e' esso
#    stesso dentro la cartella protetta. Quindi `arm` COPIA il tick in
#    ~/.thesis_vm_retry e `status` confronta lo sha256 della copia con
#    l'originale versionato: se divergono, lo dice.
# =====================================================================
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LABEL=com.leonardo.thesis.vmstart
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
RUNTIME="$HOME/.thesis_vm_retry"
SRC="$REPO/scripts/vm_start_retry.sh"
TICK="$RUNTIME/vm_start_retry.sh"
GCLOUD=${THESIS_GCLOUD_BIN:-/opt/homebrew/bin/gcloud}
INTERVAL=${THESIS_VM_RETRY_INTERVAL:-600}
ACTION=${1:-status}

case "$ACTION" in
  arm)
    HOURS=${2:-24}
    mkdir -p "$RUNTIME"
    cp "$SRC" "$TICK" && chmod 755 "$TICK"
    rm -f "$RUNTIME/stop"
    echo 0 > "$RUNTIME/attempts"
    echo $(( $(date +%s) + HOURS * 3600 )) > "$RUNTIME/deadline"
    cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${TICK}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>THESIS_VM_RETRY_STATE</key><string>${RUNTIME}</string>
    <key>THESIS_GCLOUD_BIN</key><string>${GCLOUD}</string>
  </dict>
  <key>StartInterval</key><integer>${INTERVAL}</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>${RUNTIME}/launchd.out</string>
  <key>StandardErrorPath</key><string>${RUNTIME}/launchd.err</string>
</dict>
</plist>
PLIST_EOF
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" "$PLIST" || {
      echo "🛑 launchctl bootstrap ha fallito — il loop NON e' armato"; exit 1; }
    echo "armato: un tentativo ogni ${INTERVAL}s, scadenza fra ${HOURS} h"
    echo "scadenza: $(date -u -r "$(cat "$RUNTIME/deadline")" +%Y-%m-%dT%H:%M:%SZ)"
    echo "tick    : $TICK"
    echo "log     : $RUNTIME/retry.log"
    ;;

  status)
    echo "--- launchd ---"
    if launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1; then
      launchctl print "gui/$(id -u)/${LABEL}" \
        | grep -E "^[[:space:]]*(state|runs|last exit code) =" | sed 's/^[[:space:]]*/  /'
    else
      echo "  NON caricato"
    fi
    echo "--- la copia che gira e' quella versionata? ---"
    if [ -f "$TICK" ]; then
      A=$(shasum -a 256 "$SRC" | cut -d' ' -f1)
      B=$(shasum -a 256 "$TICK" | cut -d' ' -f1)
      if [ "$A" = "$B" ]; then
        echo "  si': sha256 ${A:0:12}…"
      else
        echo "  🛑 NO — la copia in esecuzione differisce dall'originale nel repo."
        echo "     repo: ${A:0:12}…  copia: ${B:0:12}…   (ri-arma per allineare)"
      fi
    else
      echo "  nessuna copia installata"
    fi
    echo "--- stato del ritentativo ---"
    echo "  tentativi : $(cat "$RUNTIME/attempts" 2>/dev/null || echo '-')"
    [ -f "$RUNTIME/deadline" ] && \
      echo "  scadenza  : $(date -u -r "$(cat "$RUNTIME/deadline")" +%Y-%m-%dT%H:%M:%SZ)"
    if [ -f "$RUNTIME/stop" ]; then
      echo "  FERMO     : $(cat "$RUNTIME/stop")"
    else
      echo "  FERMO     : no (sta ancora riprovando)"
    fi
    echo "--- ultime righe ---"
    tail -8 "$RUNTIME/retry.log" 2>/dev/null | sed 's/^/  /'
    if [ -s "$RUNTIME/launchd.err" ]; then
      echo "--- launchd stderr (NON deve contenere nulla) ---"
      tail -4 "$RUNTIME/launchd.err" | sed 's/^/  /'
    fi
    ;;

  disarm)
    launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null
    mkdir -p "$RUNTIME"
    echo "disarmato a mano" > "$RUNTIME/stop"
    rm -f "$PLIST"
    echo "disarmato: agent rimosso e file stop scritto"
    ;;

  *)
    echo "uso: $0 {arm [ORE]|status|disarm}" >&2
    exit 2
    ;;
esac
