#!/usr/bin/env bash
# Cell-border harness: launch minimized off a pinned save -> let the AutoWalk lua mod glide the player
# across cell borders -> report what the grid-driven texture/geometry systems did at each crossing.
# Usage: forge-walk-run.sh [delay=12] [save=ibodragonstareast360.ess] [speed=600] [distance=24576]
#                          [heading=-1] [timeout=300] [clientEnv="NAME=v,NAME=v"] [trace] [settle=20]
#
# heading: degrees, 0 = north, 90 = east; negative = the save's own facing. distance 24576 = three
# cells, so at least two border crossings whatever the start position. settle defaults past the
# 600-frame stale-texture eviction age: a release of anything still in the grid shows up in the log.
# clientEnv / trace as in forge-popin-run.sh (trace -> MGE_FRAME_TRACE=1; read it with
# mgexe-devkit/tools/frametrace-frames.py).
#
# THE MEASUREMENT (mgeXE.log unless noted):
#   [autowalk] (MWSE.log)  the ground truth: every cell MW entered, and whether the writes took.
#   [tex-prefetch] grid    one per grid move: how many of the new grid's textures were still to load.
#   [tex-evict] released   what left with the dropped row. Must not name the grid you stand in.
#   [hb] tex residency     slots, recycles (must stay 0), evictions.
#   per-frame dt            from the trace: a crossing that hitches shows as a cluster of slow frames.
set -u
DELAY="${1:-12}"
SAVE="${2:-ibodragonstareast360.ess}"
SPEED="${3:-600}"
DISTANCE="${4:-24576}"
HEADING="${5:--1}"
TIMEOUT="${6:-300}"
CLIENTENV="${7:-}"
TRACE="${8:-}"
SETTLE="${9:-20}"
CLIENT_ENV_NAMES="MGE_TIER1_SEM MGE_TIER1_EVENT MGE_COPY_AT_BLIT MGE_FRAME_AHEAD MGE_TEX_PREFETCH_MB MGE_TEX_STREAM_MB MGE_TEX_EVICT_FRAMES MGE_TEX_BOOKKEEP MGE_GEOM_CONTENT_PROBE MGE_GEOM_ALIAS MGE_GEOM_ALIAS_SKIN MGE_GEOM_MEMO MGE_GEOM_KEEP_MB MGE_TEX_STREAM_ASYNC MGE_TEX_IO_THREAD MGE_SCOPED_WALK MGE_GATE_LAZY MGE_WALK_RESUME MGE_KEEP_EXTERIOR"

MW="/mnt/c/mgem/morrowind64"
XELOG="$MW/mgeXE.log"
MWSELOG="$MW/MWSE.log"
ILCFG="$MW/Data Files/MWSE/config/instant load.json"
AWCFG="$MW/Data Files/MWSE/config/AutoWalk.json"

if [ ! -f "$MW/Saves/$SAVE" ]; then
  echo "[walk] ERROR: save not found: Saves/$SAVE" >&2
  exit 1
fi
if [ ! -f "$MW/Data Files/MWSE/mods/AutoWalk/main.lua" ]; then
  echo "[walk] ERROR: AutoWalk mod not deployed to $MW/Data Files/MWSE/mods/ (from tools/mwse-dev-mods/)" >&2
  exit 1
fi

ILBAK="$(mktemp)"
cp "$ILCFG" "$ILBAK" 2>/dev/null
# Restore the instant-load config and DELETE the AutoWalk one however we exit: left armed, it would
# drag the player across the map on every load of a normal play session.
restore() {
  cp "$ILBAK" "$ILCFG" 2>/dev/null
  rm -f "$ILBAK" "$AWCFG"
  echo "[walk] restored instant-load config, removed AutoWalk config"
}
trap restore EXIT

python3 - "$ILCFG" "$SAVE" "$AWCFG" "$DELAY" "$SPEED" "$DISTANCE" "$HEADING" "$SETTLE" <<'PY'
import json, sys
il, save, aw, delay, speed, distance, heading, settle = sys.argv[1:9]
with open(il) as f: cfg = json.load(f)
cfg["continue"] = False          # else the mod loads the NEWEST save, not ours
cfg["overrideFile"] = save
with open(il, "w") as f: json.dump(cfg, f, indent=2)
with open(aw, "w") as f:
    json.dump({"enabled": True, "delay": float(delay), "speed": float(speed),
               "distance": float(distance), "heading": float(heading), "settle": float(settle)}, f, indent=2)
PY
echo "[walk] pinned save=$SAVE | delay=${DELAY}s speed=$SPEED distance=$DISTANCE heading=$HEADING settle=${SETTLE}s"

powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
for _ in $(seq 1 20); do
  alive=$(powershell.exe -Command "@(Get-Process Morrowind,mgeHost64 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  [ "${alive:-0}" = "0" ] && break
  sleep 1
done

ARCHIVE="$MW/logarchive"
mkdir -p "$ARCHIVE"
stamp=$(date +%Y%m%d-%H%M%S)
for f in "$XELOG" "$MW/mgeHost64.log" "$MWSELOG"; do
  [ -s "$f" ] && cp "$f" "$ARCHIVE/$(basename "$f" .log)-$stamp.log" 2>/dev/null
done
ls -1t "$ARCHIVE"/mgeXE-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/MWSE-*.log  2>/dev/null | tail -n +21 | xargs -r rm -f

ENVSET="Remove-Item Env:MGE_RDOC -ErrorAction SilentlyContinue; \$env:MGE_AUTODISMISS='1'; "
ENVSET="${ENVSET}Remove-Item Env:MGE_EXACT_POS -ErrorAction SilentlyContinue; "
if [ -n "$TRACE" ]; then
  echo "[walk] MGE_FRAME_TRACE=1"
  ENVSET="${ENVSET}\$env:MGE_FRAME_TRACE='1'; "
else
  ENVSET="${ENVSET}Remove-Item Env:MGE_FRAME_TRACE -ErrorAction SilentlyContinue; "
fi
ENVSET="${ENVSET}Remove-Item Env:MGE_HOST_KNOBS -ErrorAction SilentlyContinue; "
for n in $CLIENT_ENV_NAMES; do
  ENVSET="${ENVSET}Remove-Item Env:$n -ErrorAction SilentlyContinue; "
done
if [ -n "$CLIENTENV" ]; then
  echo "[walk] client env = $CLIENTENV"
  IFS=',' read -r -a _pairs <<< "$CLIENTENV"
  for p in "${_pairs[@]}"; do
    ENVSET="${ENVSET}\$env:${p%%=*}='${p#*=}'; "
  done
fi
powershell.exe -Command "${ENVSET}Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
echo "[walk] launched Morrowind; waiting for the walk..."

t0=$(date +%s)
while :; do
  sleep 3
  el=$(( $(date +%s) - t0 ))
  settled=$(grep -c "\[autowalk\] SETTLED" "$MWSELOG" 2>/dev/null; true)
  done_=$(grep -c "\[autowalk\] WALK DONE" "$MWSELOG" 2>/dev/null; true)
  nproc=$(powershell.exe -Command "@(Get-Process Morrowind -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  running=$([ "${nproc:-0}" -gt 0 ] && echo True || echo False)
  echo "[walk] t=${el}s walkDone=$done_ settled=$settled running=$running"
  if [ "${settled:-0}" -ge 1 ]; then echo "[walk] walk + settle complete"; break; fi
  if [ "$running" = "False" ]; then echo "[walk] Morrowind EXITED before settling (crash?)"; break; fi
  if [ "$el" -ge "$TIMEOUT" ]; then echo "[walk] TIMEOUT after ${el}s"; break; fi
done

echo
echo "=== VERIFY (device removed / FAILED / fatal) ==="
grep -Ei "device removed|FAILED|fatal|crash" "$MW/mgeHost64.log" 2>/dev/null | tail -10 || true
echo "  (end)"

echo
echo "=== autowalk timeline (MWSE.log) ==="
grep "\[autowalk\]" "$MWSELOG" 2>/dev/null || echo "  !! no [autowalk] lines - mod did not run"

echo
echo "=== grid-driven texture work (mgeXE.log) ==="
grep -E "\[tex-prefetch\] grid|\[tex-evict\] released|working set exceeds|\[cell-purge\]" "$XELOG" 2>/dev/null || echo "  (none)"

echo
echo "=== tex residency heartbeats ==="
grep -E "\[hb\] tex residency" "$XELOG" 2>/dev/null | sed -E 's/^>> \[hb\] tex residency: /  /'

echo
echo "=== frame heartbeats (dt / worst frame) ==="
grep -E "^>> \[hb\] [0-9]+ frames" "$XELOG" 2>/dev/null | sed -E 's/.*(dt=[0-9.]+).*(max feed=[0-9.]+ dt=[0-9.]+ \(~[0-9]+ fps\))/  \1 \2/'

echo
echo "[walk] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[walk] done"
