#!/usr/bin/env bash
# Forge automated perf harness: launch minimized (auto-loads test scene) -> poll host log for N new
# 'gpu split' heartbeats -> verify no device-removal -> kill both procs -> report the last N splits.
# Usage: forge-perf-run.sh [samples=5] [timeout=180] [save.ess]
#
# THE SAVE ARGUMENT MATTERS. Without it the "instant load" mod runs in continue=true mode and loads
# whatever .ess is NEWEST — i.e. the measured scene is whatever you last saved, which silently
# invalidates any cross-run comparison. (Cost us once: a shadow-mask cost recorded in a light-dense
# interior did not reproduce, because the harness had drifted onto an exterior save.) Passing a save
# pins it via the mod's own overrideFile config; the previous config is restored on exit.
set -u
SAMPLES="${1:-5}"
TIMEOUT="${2:-180}"
SAVE="${3:-}"
LOG="/mnt/c/mgem/morrowind64/mgeHost64.log"
CFG="/mnt/c/mgem/morrowind64/Data Files/MWSE/config/instant load.json"
CFGBAK="$(mktemp)"

if [ -n "$SAVE" ]; then
  if [ ! -f "/mnt/c/mgem/morrowind64/Saves/$SAVE" ]; then
    echo "[harness] ERROR: save not found: Saves/$SAVE" >&2
    exit 1
  fi
  cp "$CFG" "$CFGBAK"
  python3 - "$CFG" "$SAVE" <<'PY'
import json, sys
path, save = sys.argv[1], sys.argv[2]
with open(path) as f: cfg = json.load(f)
cfg["continue"] = False          # else the mod loads the NEWEST save, not ours
cfg["overrideFile"] = save
with open(path, "w") as f: json.dump(cfg, f, indent=2)
PY
  echo "[harness] pinned save = $SAVE"
  # Always put the user's config back, even on ctrl-C / timeout / crash — leaving it pinned would
  # hijack normal play.
  trap 'cp "$CFGBAK" "$CFG"; rm -f "$CFGBAK"; echo "[harness] restored instant-load config"' EXIT
else
  echo "[harness] WARNING: no save pinned — loading the NEWEST .ess (scene is whatever you saved last)"
fi

# Make sure no PREVIOUS run is still alive. Back-to-back sweep runs raced this: a second Morrowind
# launched while the first was still shutting down, exited immediately, and the sweep then copied the
# PREVIOUS scene's log as this scene's result — a silently wrong row, which is worse than a missing
# one. Kill and wait for the handles to actually go before touching the log offset.
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
for _ in $(seq 1 20); do
  alive=$(powershell.exe -Command "@(Get-Process Morrowind,mgeHost64 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  [ "${alive:-0}" = "0" ] && break
  sleep 1
done

# Archive whatever is in the logs before launching. The host TRUNCATES mgeHost64.log at startup, so
# a harness run silently destroys the log of whatever came before it — including a play session the
# user has just reported a bug from. Cost is a file copy; the alternative is unreproducible evidence.
ARCHIVE="/mnt/c/mgem/morrowind64/logarchive"
mkdir -p "$ARCHIVE"
stamp=$(date +%Y%m%d-%H%M%S)
for f in "$LOG" /mnt/c/mgem/morrowind64/mgeXE.log; do
  [ -s "$f" ] && cp "$f" "$ARCHIVE/$(basename "$f" .log)-$stamp.log" 2>/dev/null
done
# Keep the 20 most recent of each; these run to tens of MB.
ls -1t "$ARCHIVE"/mgeHost64-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/mgeXE-*.log     2>/dev/null | tail -n +21 | xargs -r rm -f

startlines=0
[ -f "$LOG" ] && startlines=$(wc -l < "$LOG")
echo "[harness] start offset = $startlines lines; want $SAMPLES new 'gpu split' samples (timeout ${TIMEOUT}s)"

# Launch minimized (no focus steal). The save auto-loads.
powershell.exe -Command "Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
echo "[harness] launched Morrowind; polling..."

t0=$(date +%s)
got=0
while :; do
  sleep 3
  now=$(date +%s); el=$((now - t0))
  cur=0; [ -f "$LOG" ] && cur=$(wc -l < "$LOG")
  # The host TRUNCATES mgeHost64.log on launch → cur < startlines means the log rotated; measure from 0.
  if [ "$cur" -lt "$startlines" ]; then startlines=0; fi
  if [ "$cur" -gt "$startlines" ]; then
    got=$(tail -n +$((startlines + 1)) "$LOG" | grep -c "gpu split:")
  fi
  # crash guard: Morrowind gone with no samples
  # @(...).Count, not "-ne $null": with two Morrowind handles alive the latter formats BOTH process
  # objects into the output and the comparison silently becomes garbage.
  nproc=$(powershell.exe -Command "@(Get-Process Morrowind -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  running=$([ "${nproc:-0}" -gt 0 ] && echo True || echo False)
  echo "[harness] t=${el}s newlines=$((cur-startlines)) gpuSplits=$got running=$running"
  if [ "$got" -ge "$SAMPLES" ]; then echo "[harness] got $got samples"; break; fi
  if [ "$running" = "False" ] && [ "$got" -eq 0 ]; then echo "[harness] Morrowind EXITED with 0 samples (crash?)"; break; fi
  if [ "$el" -ge "$TIMEOUT" ]; then echo "[harness] TIMEOUT after ${el}s ($got samples)"; break; fi
done

echo "=== VERIFY (device removed / FAILED / fatal in new log) ==="
tail -n +$((startlines + 1)) "$LOG" | grep -Ei "device removed|FAILED|fatal|crash" | tail -20 || echo "  (clean)"

echo "=== last splits ==="
tail -n +$((startlines + 1)) "$LOG" | grep -E "host split:|gpu split:|gpu color sub:|\[dl\] exterior|dist lights " | tail -$((SAMPLES * 4))

echo "[harness] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[harness] done"
