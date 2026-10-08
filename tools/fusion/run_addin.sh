#!/usr/bin/env bash
# run_addin.sh <add-in folder> <result file under fusion_jobs> [document name whose cloud upload to wait for]
# One-shot add-in run: (re)start Fusion with the add-in, wait for its result file, remove the add-in again and, if the
# result says "ok", wait for the named document's upload. Refuses to close a Fusion that has unsaved changes, or one the
# user touched in the last 2 minutes (a closed Fusion can be started any time).
set -u
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority
AU=$HOME/.local/share/Autodesk-Unofficial
PFX=$AU/wineprefixes/fusion-1
AD="$PFX/drive_c/users/thiago/AppData/Roaming/Autodesk/Autodesk Fusion 360/API/AddIns"
J="$PFX/drive_c/fusion_jobs"
LOGS="$PFX/drive_c/users/thiago/AppData/Local/Autodesk/Autodesk Fusion 360/U2X2EAHRKTHP3PTF/logs"
running() { ps -eo args | grep -qE 'Fusion36[0]\.exe'; }
if running; then
  idle=$(DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus gdbus call --session --dest org.gnome.Mutter.IdleMonitor \
         --object-path /org/gnome/Mutter/IdleMonitor/Core --method org.gnome.Mutter.IdleMonitor.GetIdletime | sed -E 's/.*uint64 ([0-9]+).*/\1/')
  [ "${idle:-0}" -lt 120000 ] && { echo "USER ACTIVE (idle ${idle} ms) - not restarting Fusion"; exit 3; }
  T=$(wmctrl -l | grep -E 'Autodesk Fusion' | cut -d' ' -f5-)
  echo "$T" | grep -q '\*' && { echo "UNSAVED CHANGES in Fusion ($T) - not restarting"; exit 4; }
  for id in $(wmctrl -l | grep -E 'Unable to sign in|Autodesk Fusion' | awk '{print $1}'); do wmctrl -i -c "$id"; sleep 1; done
  for i in $(seq 1 60); do running || break; sleep 2; done
  running && { WINEPREFIX=$PFX "$AU/fusion-wine-build/bin/wineserver" -k; sleep 3; echo "forced stop"; }
fi
rm -f "$J/$2"; rm -rf "$AD/$(basename "$1")"; cp -r "$1" "$AD/"
setsid bash -c '"$HOME/.local/share/Autodesk-Unofficial/bin/autodesk_fusion_launcher.sh" fusion' </dev/null >/dev/null 2>&1 &
t=0; until [ -f "$J/$2" ] || [ $t -gt 3000 ]; do sleep 10; t=$((t + 10)); done
sleep 2; rm -rf "$AD/$(basename "$1")"
[ -f "$J/$2" ] || { echo "no result after ${t}s"; exit 5; }
if [ -n "${3:-}" ] && grep -q '"status": "ok"' "$J/$2"; then   # only a saved result uploads
  L=$(ls -t "$LOGS"/AppLogFile*.log | head -1)
  for i in $(seq 1 120); do grep -aq "Uploaded document $3," "$L" && { echo "upload done"; break; }; sleep 10; done
fi
echo "result: $J/$2"
