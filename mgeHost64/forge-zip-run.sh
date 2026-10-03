#!/usr/bin/env bash
# Exterior cell-change STRESS: launch minimized off a pinned save -> let the AutoZip lua mod teleport
# (far jumps) and glide (fast grid slides) the player through `target` cell changes -> report whether
# anything leaked, backed up, went white or fell over.
# Usage: forge-zip-run.sh [delay=12] [save=ibodragonstareast360.ess] [target=500] [seed=1]
#                         [timeout=1800] [clientEnv="NAME=v,NAME=v"] [trace] [farChance=0.4] [settle=20]
#                         [mode=zip|hop] [hops=10] [dwell=4]
#
# mode=hop is the DOOR HOP: `hops` round trips exterior -> "Seyda Neen, Census and Excise Office" ->
# the same exterior spot, `dwell` real seconds each side; MWSE.log logs each leg's positionCell ms.
#
# THE MEASUREMENT — trends, not events: a stress run passes when the last heartbeats look like the
# first ones.
#   [autozip] (MWSE.log)   ground truth: cell changes done, legs far/glide, and why it stopped.
#   crash / device removal Morrowind must still be running at SETTLED; host log clean.
#   [hb] mw mem            client process memory (32-bit: the ceiling that bites first).
#   [hb] tex residency     slots / resident / recycles (must stay 0) / evictions.
#   [mem] (host)           host VRAM + arena shape.
#   white / thrash / out of memory / release backlog lines: must be absent or bounded.
set -u
DELAY="${1:-12}"
SAVE="${2:-ibodragonstareast360.ess}"
TARGET="${3:-500}"
SEED="${4:-1}"
TIMEOUT="${5:-1800}"
CLIENTENV="${6:-}"
TRACE="${7:-}"
FARCHANCE="${8:-0.4}"
SETTLE="${9:-20}"
MODE="${10:-zip}"
HOPS="${11:-10}"
DWELL="${12:-4}"
CLIENT_ENV_NAMES="MGE_TIER1_SEM MGE_TIER1_EVENT MGE_COPY_AT_BLIT MGE_FRAME_AHEAD MGE_TEX_PREFETCH_MB MGE_TEX_STREAM_MB MGE_TEX_EVICT_FRAMES MGE_TEX_BOOKKEEP MGE_GEOM_CONTENT_PROBE MGE_GEOM_ALIAS MGE_GEOM_KEEP_MB MGE_TEX_STREAM_ASYNC MGE_TEX_IO_THREAD MGE_SCOPED_WALK MGE_GATE_LAZY"

MW="/mnt/c/mgem/morrowind64"
XELOG="$MW/mgeXE.log"
HOSTLOG="$MW/mgeHost64.log"
MWSELOG="$MW/MWSE.log"
ILCFG="$MW/Data Files/MWSE/config/instant load.json"
AZCFG="$MW/Data Files/MWSE/config/AutoZip.json"

if [ ! -f "$MW/Saves/$SAVE" ]; then
  echo "[zip] ERROR: save not found: Saves/$SAVE" >&2
  exit 1
fi
if [ ! -f "$MW/Data Files/MWSE/mods/AutoZip/main.lua" ]; then
  echo "[zip] ERROR: AutoZip mod not deployed to $MW/Data Files/MWSE/mods/ (from tools/mwse-dev-mods/)" >&2
  exit 1
fi

ILBAK="$(mktemp)"
cp "$ILCFG" "$ILBAK" 2>/dev/null
# Left armed, AutoZip would teleport the player around the map on every load of a normal session.
restore() {
  cp "$ILBAK" "$ILCFG" 2>/dev/null
  rm -f "$ILBAK" "$AZCFG"
  echo "[zip] restored instant-load config, removed AutoZip config"
}
trap restore EXIT

python3 - "$ILCFG" "$SAVE" "$AZCFG" "$DELAY" "$TARGET" "$SEED" "$FARCHANCE" "$SETTLE" "$MODE" "$HOPS" "$DWELL" <<'PY'
import json, sys
il, save, az, delay, target, seed, far, settle, mode, hops, dwell = sys.argv[1:12]
with open(il) as f: cfg = json.load(f)
cfg["continue"] = False          # else the mod loads the NEWEST save, not ours
cfg["overrideFile"] = save
with open(il, "w") as f: json.dump(cfg, f, indent=2)
with open(az, "w") as f:
    json.dump({"enabled": True, "delay": float(delay), "target": int(target), "seed": int(seed),
               "farChance": float(far), "settle": float(settle), "mode": mode, "hops": int(hops),
               "dwell": float(dwell)}, f, indent=2)
PY
echo "[zip] pinned save=$SAVE | mode=$MODE delay=${DELAY}s target=$TARGET seed=$SEED farChance=$FARCHANCE settle=${SETTLE}s hops=$HOPS dwell=${DWELL}s"

powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
for _ in $(seq 1 20); do
  alive=$(powershell.exe -Command "@(Get-Process Morrowind,mgeHost64 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  [ "${alive:-0}" = "0" ] && break
  sleep 1
done

ARCHIVE="$MW/logarchive"
mkdir -p "$ARCHIVE"
stamp=$(date +%Y%m%d-%H%M%S)
for f in "$XELOG" "$HOSTLOG" "$MWSELOG"; do
  [ -s "$f" ] && cp "$f" "$ARCHIVE/$(basename "$f" .log)-$stamp.log" 2>/dev/null
done
ls -1t "$ARCHIVE"/mgeXE-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/MWSE-*.log  2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/mgeHost64-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f

ENVSET="Remove-Item Env:MGE_RDOC -ErrorAction SilentlyContinue; \$env:MGE_AUTODISMISS='1'; "
ENVSET="${ENVSET}Remove-Item Env:MGE_EXACT_POS -ErrorAction SilentlyContinue; "
if [ -n "$TRACE" ]; then
  echo "[zip] MGE_FRAME_TRACE=1"
  ENVSET="${ENVSET}\$env:MGE_FRAME_TRACE='1'; "
else
  ENVSET="${ENVSET}Remove-Item Env:MGE_FRAME_TRACE -ErrorAction SilentlyContinue; "
fi
ENVSET="${ENVSET}Remove-Item Env:MGE_HOST_KNOBS -ErrorAction SilentlyContinue; "
for n in $CLIENT_ENV_NAMES; do
  ENVSET="${ENVSET}Remove-Item Env:$n -ErrorAction SilentlyContinue; "
done
if [ -n "$CLIENTENV" ]; then
  echo "[zip] client env = $CLIENTENV"
  IFS=',' read -r -a _pairs <<< "$CLIENTENV"
  for p in "${_pairs[@]}"; do
    ENVSET="${ENVSET}\$env:${p%%=*}='${p#*=}'; "
  done
fi
powershell.exe -Command "${ENVSET}Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
echo "[zip] launched Morrowind; waiting for the run..."

t0=$(date +%s); last_report=0; alive_at_end=0
while :; do
  sleep 5
  el=$(( $(date +%s) - t0 ))
  settled=$(grep -c "\[autozip\] SETTLED" "$MWSELOG" 2>/dev/null; true)
  changes=$(grep -c "\[autozip\] #" "$MWSELOG" 2>/dev/null; true)
  nproc=$(powershell.exe -Command "@(Get-Process Morrowind -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  running=$([ "${nproc:-0}" -gt 0 ] && echo True || echo False)
  if [ $(( el - last_report )) -ge 30 ]; then
    echo "[zip] t=${el}s cell changes=$changes running=$running"
    last_report=$el
  fi
  if [ "${settled:-0}" -ge 1 ]; then alive_at_end=1; echo "[zip] run + settle complete at t=${el}s"; break; fi
  if [ "$running" = "False" ]; then echo "[zip] !! Morrowind EXITED at t=${el}s after $changes cell changes (crash?)"; break; fi
  if [ "$el" -ge "$TIMEOUT" ]; then echo "[zip] !! TIMEOUT after ${el}s ($changes cell changes)"; break; fi
done

echo
echo "=== autozip (MWSE.log) ==="
grep -E "\[autozip\] (armed|ZIP|HOP|hop |SETTLED|!!)" "$MWSELOG" 2>/dev/null || echo "  !! no [autozip] summary lines"
grep -iE "error|traceback" "$MWSELOG" 2>/dev/null | grep -v "\[autozip\]" | head -10

echo
echo "=== host health (device removed / FAILED / fatal / forge-log errors) ==="
grep -Ei "device removed|FAILED|fatal|crash|!! \[forge-log\]" "$HOSTLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -15
echo "  (end)"

echo
echo "=== client trouble lines (counts) ==="
grep -oE "!! \[[a-z0-9 -]+\][^0-9(]{0,40}" "$XELOG" 2>/dev/null | sort | uniq -c | sort -rn | head -25

echo
echo "=== purges / prefetch / eviction (counts) ==="
echo "  cell purges:        $(grep -c "\[cell-purge\]" "$XELOG" 2>/dev/null)"
echo "  post-load windows:  $(grep -c "\[postload-walk\] exterior done" "$XELOG" 2>/dev/null)"
echo "  prefetch scans:     $(grep -c "\[tex-prefetch\] grid (" "$XELOG" 2>/dev/null)"
echo "  eviction passes:    $(grep -c "\[tex-evict\] released" "$XELOG" 2>/dev/null)"

echo
echo "=== trends: first / middle / last heartbeats ==="
python3 - "$XELOG" "$HOSTLOG" <<'PY'
import re, sys
xe = open(sys.argv[1], errors="ignore").read().splitlines()
try:
    host = open(sys.argv[2], errors="ignore").read().splitlines()
except OSError:
    host = []

def pick(lines, label):
    if not lines:
        print(f"  {label}: (none)"); return
    idx = sorted({0, len(lines) // 2, len(lines) - 1})
    print(f"  {label} ({len(lines)} heartbeats):")
    for i in idx:
        print(f"    [{i:4d}] {lines[i][:230]}")

pick([l for l in xe if "[hb] mw mem" in l], "client memory")
pick([l for l in xe if "[hb] tex residency" in l], "texture residency")
pick([l for l in xe if re.match(r"^>> \[gc\] \d+ frames", l)], "geometry cache")
pick([l for l in host if "[mem]" in l], "host memory")

dts = [float(m.group(1)) for l in xe if (m := re.search(r"\[hb\] \d+ frames avg:.*\| max feed=[\d.]+ dt=([\d.]+)", l))]
avg = [float(m.group(1)) for l in xe if (m := re.search(r"\[hb\] \d+ frames avg:.* dt=([\d.]+) mwstart", l))]
if dts:
    s = sorted(dts)
    print(f"  worst frame per 300-frame window: median {s[len(s)//2]:.1f} ms, p90 {s[int(len(s)*0.9)]:.1f}, max {s[-1]:.1f} ({len(s)} windows)")
if avg:
    print(f"  avg dt per window: first {avg[0]:.2f}, last {avg[-1]:.2f}, max {max(avg):.2f} ms")
PY

echo
echo "[zip] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[zip] done (alive at end: $alive_at_end)"
