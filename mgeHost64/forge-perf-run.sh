#!/usr/bin/env bash
# Forge automated perf harness: launch minimized (auto-loads test scene) -> poll host log for N new
# 'gpu split' heartbeats -> verify no device-removal -> kill both procs -> report the last N splits.
# Usage: forge-perf-run.sh [samples=5] [timeout=180] [save.ess] [renderScale]
#
# renderScale (4th arg, 1.0-2.0) drives the client's MGE_RENDER_SCALE startup override, which is the
# ONLY scriptable way to change the internal render resolution — the live knob is a panel slider and
# nobody is at the keyboard during a minimized run. Omitted => whatever the build defaults to (1.0).
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
SCALE="${4:-}"
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
# Same offset trick on the CLIENT log. The host's numbers alone cannot answer "is the host the
# bottleneck" — only the client's render=[host=] / overlap= pair says how much of the host frame the
# client actually waited for. mgecore does NOT truncate mgeXE.log, so this offset is what separates
# this run from the session before it.
CLOG="/mnt/c/mgem/morrowind64/mgeXE.log"
cstartlines=0
[ -f "$CLOG" ] && cstartlines=$(wc -l < "$CLOG")
echo "[harness] start offset = $startlines lines; want $SAMPLES new 'gpu split' samples (timeout ${TIMEOUT}s)"

# Launch minimized (no focus steal). The save auto-loads.
# The render-scale override has to be set INSIDE the same powershell that calls Start-Process:
# exporting it from bash does not cross the WSL->Win32 boundary without WSLENV, and a var that
# silently fails to arrive would report the wrong resolution's numbers under the right label.
# MGE_RDOC makes main.cpp LoadLibrary renderdoc.dll BEFORE device creation, so the measured host
# runs with RenderDoc hooking every D3D12 call. That is a perf confound of unknown, probably
# non-uniform size (same class as the EcoQoS throttle above), and its crash handler swallows faults
# into a modal dialog no one can click during a minimized run. It was left set as a persistent USER
# variable on 2026-08-04 and silently rode along in every measurement for three days. Stripped HERE
# rather than trusted to the ambient environment, because clearing the registry value does NOT fix
# an already-running WSL session: interop hands Windows children a cached env block, so the stale
# MGE_RDOC=1 keeps arriving until WSL restarts. Set it deliberately if you want a capture.
RDOC_STRIP="Remove-Item Env:MGE_RDOC -ErrorAction SilentlyContinue; "
if [ -n "$SCALE" ]; then
  echo "[harness] render scale = ${SCALE}x (MGE_RENDER_SCALE)"
  powershell.exe -Command "${RDOC_STRIP}\$env:MGE_RENDER_SCALE='$SCALE'; Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
else
  powershell.exe -Command "${RDOC_STRIP}Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
fi
echo "[harness] launched Morrowind; polling..."

t0=$(date +%s)
got=0
unthrottled=0
while :; do
  sleep 3
  # Undo Windows' background/minimized power throttling as soon as the host exists. Applied ONCE,
  # not every poll: each call spawns a powershell, and the setting is sticky for the process
  # lifetime. Skipping this cost ~2.2x on every number in the run — see forge-perf-unthrottle.ps1.
  if [ "$unthrottled" = "0" ]; then
    hostalive=$(powershell.exe -Command "@(Get-Process mgeHost64 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
    if [ "${hostalive:-0}" -gt 0 ]; then
      powershell.exe -ExecutionPolicy Bypass -File 'C:\projects\mgexe\MGE-XE\mgeHost64\forge-perf-unthrottle.ps1' 2>&1 | sed 's/^/[harness] /'
      unthrottled=1
    fi
  fi
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

# The client side of the same frames. [seam] backbuffer is printed FIRST so every table row carries
# the resolution it was actually measured at, rather than the one that was asked for.
echo "=== client (mgeXE.log) ==="
if [ "$(wc -l < "$CLOG" 2>/dev/null || echo 0)" -lt "$cstartlines" ]; then cstartlines=0; fi
tail -n +$((cstartlines + 1)) "$CLOG" 2>/dev/null \
  | grep -E "\[seam\] backbuffer|MGE_RENDER_SCALE|\[hb\] [0-9]+ frames avg:|\[hb\] host recv:|\[produce\] overlap:" \
  | tail -12 || echo "  (no client heartbeats)"

echo "[harness] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[harness] done"
