#!/usr/bin/env bash
# Post-load pop-in harness: launch minimized off a pinned save -> let the AutoTurn360 lua mod sweep
# the player through 360 deg -> report whether any geometry was captured FIRST-SIGHT during the turn.
# Usage: forge-popin-run.sh [delay=0] [save=playeroldebonheart.ess] [duration=3] [reload=0] [timeout=240]
#
# reload=1 is THE repro (c). A first load runs its loading frames before the seam owns them, so the
# full refresh walk still runs and pre-populates the cache - measured 11705 entries already cached at
# the first [cell-purge], which is why a first-load turn discovers almost nothing. A RELOAD has the
# host already up: those frames are liveDrawBuild, walk nothing, and checkCellEpochAndPurge purges to
# EMPTY, so the cache refills frustum-only. Same path as a door transition or fast travel.
#
# THE MEASUREMENT. Pop-in is "geometry arrived while I was looking at it". The client log answers
# that directly, no eyeballs needed:
#   [postload-walk]  the residency window's own receipt: cache=A->B captures=N before the turn.
#   [gc] entries=    cache size. FLAT across the turn = everything was already resident.
#                    Climbing = the turn is discovering geometry = pop-in.
#   [gc] cap=        first-sight lazy captures per frame. ~0 across the turn is the pass condition.
#   [hb] maxBuild=   the burst tell: a capture burst lands as one fat build, which is the hitch.
#   [hb] max dt=     worst frame in the window - the "slowdown as I rotate" the report describes.
#
# delay=0 is repro (a) "load and turn immediately"; delay=12 is repro (b) "wait, then turn" (past
# the 600-frame kCaptureEpochGraceFrames, which is the worse case because the capture budget has
# dropped back to 32/frame). Both configs are restored on exit - a left-armed AutoTurn360 would take
# the controls away during normal play.
set -u
DELAY="${1:-0}"
SAVE="${2:-playeroldebonheart.ess}"
DURATION="${3:-3}"
RELOAD="${4:-0}"
TIMEOUT="${5:-240}"

MW="/mnt/c/mgem/morrowind64"
XELOG="$MW/mgeXE.log"
MWSELOG="$MW/MWSE.log"
ILCFG="$MW/Data Files/MWSE/config/instant load.json"
ATCFG="$MW/Data Files/MWSE/config/AutoTurn360.json"
SETTLE=8

if [ ! -f "$MW/Saves/$SAVE" ]; then
  echo "[popin] ERROR: save not found: Saves/$SAVE" >&2
  exit 1
fi
if [ ! -f "$MW/Data Files/MWSE/mods/AutoTurn360/main.lua" ]; then
  echo "[popin] ERROR: AutoTurn360 mod not deployed to $MW/Data Files/MWSE/mods/" >&2
  exit 1
fi

ILBAK="$(mktemp)"; ATBAK="$(mktemp)"
cp "$ILCFG" "$ILBAK" 2>/dev/null
cp "$ATCFG" "$ATBAK" 2>/dev/null || echo '{}' > "$ATBAK"
# Restore BOTH configs however we exit (timeout, ctrl-C, crash). Leaving AutoTurn360 armed would
# spin the player on every load of a normal play session.
restore() {
  cp "$ILBAK" "$ILCFG" 2>/dev/null
  if [ -s "$ATBAK" ] && [ "$(cat "$ATBAK")" != "{}" ]; then cp "$ATBAK" "$ATCFG"; else rm -f "$ATCFG"; fi
  rm -f "$ILBAK" "$ATBAK"
  echo "[popin] restored instant-load + AutoTurn360 configs"
}
trap restore EXIT

python3 - "$ILCFG" "$SAVE" "$ATCFG" "$DELAY" "$DURATION" "$SETTLE" "$RELOAD" <<'PY'
import json, sys
il, save, at, delay, dur, settle, reload_ = sys.argv[1:8]
with open(il) as f: cfg = json.load(f)
cfg["continue"] = False          # else the mod loads the NEWEST save, not ours
cfg["overrideFile"] = save
with open(il, "w") as f: json.dump(cfg, f, indent=2)
with open(at, "w") as f:
    json.dump({"enabled": True, "delay": float(delay), "duration": float(dur),
               "degrees": 360, "settle": float(settle), "reload": reload_ == "1",
               "saveFile": save[:-4] if save.lower().endswith(".ess") else save}, f, indent=2)
PY
echo "[popin] pinned save=$SAVE | autoturn delay=${DELAY}s duration=${DURATION}s settle=${SETTLE}s reload=${RELOAD}"

powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
for _ in $(seq 1 20); do
  alive=$(powershell.exe -Command "@(Get-Process Morrowind,mgeHost64 -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  [ "${alive:-0}" = "0" ] && break
  sleep 1
done

# Archive the previous logs before launch (the host truncates its own; MW appends to mgeXE.log but a
# fresh run is what we want to read). Same discipline as forge-perf-run.sh.
ARCHIVE="$MW/logarchive"
mkdir -p "$ARCHIVE"
stamp=$(date +%Y%m%d-%H%M%S)
for f in "$XELOG" "$MW/mgeHost64.log" "$MWSELOG"; do
  [ -s "$f" ] && cp "$f" "$ARCHIVE/$(basename "$f" .log)-$stamp.log" 2>/dev/null
done
ls -1t "$ARCHIVE"/mgeXE-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/MWSE-*.log  2>/dev/null | tail -n +21 | xargs -r rm -f

# mgeXE.log is TRUNCATED at client init, and MWSE.log at MWSE init, so both read from 0 for this run.
powershell.exe -Command "Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
echo "[popin] launched Morrowind; waiting for the sweep..."

t0=$(date +%s); done_seen=0
while :; do
  sleep 3
  el=$(( $(date +%s) - t0 ))
  # `grep -c || echo 0` is WRONG: on zero matches grep still PRINTS "0" and then exits 1, so the
  # fallback appends a second "0" and the arithmetic test below chokes on "0\n0". grep -c always
  # emits a count; just swallow the exit status.
  settled=$(grep -c "SETTLED" "$MWSELOG" 2>/dev/null; true)
  swept=$(grep -c "SWEEP DONE" "$MWSELOG" 2>/dev/null; true)
  nproc=$(powershell.exe -Command "@(Get-Process Morrowind -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  running=$([ "${nproc:-0}" -gt 0 ] && echo True || echo False)
  echo "[popin] t=${el}s sweepDone=$swept settled=$settled running=$running"
  if [ "${settled:-0}" -ge 1 ]; then done_seen=1; echo "[popin] sweep + settle complete"; break; fi
  if [ "$running" = "False" ]; then echo "[popin] Morrowind EXITED before settling (crash?)"; break; fi
  if [ "$el" -ge "$TIMEOUT" ]; then echo "[popin] TIMEOUT after ${el}s"; break; fi
done

echo
echo "=== VERIFY (device removed / FAILED / fatal) ==="
grep -Ei "device removed|FAILED|fatal|crash" "$MW/mgeHost64.log" 2>/dev/null | tail -10 || true
echo "  (end)"

echo
echo "=== autoturn timeline (MWSE.log) ==="
grep "\[autoturn\]" "$MWSELOG" 2>/dev/null || echo "  !! no [autoturn] lines - mod did not run"

echo
echo "=== residency window (mgeXE.log) ==="
grep -E "\[cell-purge\]|\[postload-walk\]" "$XELOG" 2>/dev/null || echo "  !! none"

echo
echo "=== cache timeline: entries must be FLAT and cap ~0 across the turn ==="
grep -E "^>> \[gc\] [0-9]+ frames" "$XELOG" 2>/dev/null | sed -E 's/.*(entries=[0-9]+).*(visited=[0-9]+).*(live=[0-9]+) (cap=[0-9.]+).*/  \1 \2 \3 \4/'

echo
echo "=== build/frame timeline (burst + hitch tells) ==="
grep -E "^>> \[hb\] [0-9]+ frames|^>> \[hb\] build split" "$XELOG" 2>/dev/null | sed -E 's/.*(dt=[0-9.]+).*(max feed=[0-9.]+ dt=[0-9.]+ \(~[0-9]+ fps\))/  \1 \2/; s/.*(maxBuild=[0-9.]+ captures\/f=[0-9.]+)/  \1/'

echo
echo "=== VERDICT ==="
python3 - "$XELOG" <<'PY'
import re, sys
lines = open(sys.argv[1], errors="ignore").read().splitlines()
gc  = [l for l in lines if l.startswith(">> [gc] ") and " frames avg" in l]
ent = [int(m.group(1))   for l in gc if (m := re.search(r"entries=(\d+)",  l))]
cap = [float(m.group(1)) for l in gc if (m := re.search(r"cap=([\d.]+)",   l))]
purges = [l for l in lines if "[cell-purge]" in l]
window = [l for l in lines if "[postload-walk]" in l]
if not ent:
    print("  INCONCLUSIVE: no [gc] heartbeats in the log"); sys.exit()

print(f"  purges seen: {len(purges)} | residency windows that closed: {len(window)}")
print(f"  entries across heartbeats: {ent}")
print(f"  cap/frame  across heartbeats: {cap}")
print(f"  FINAL RESIDENT ENTRIES: {ent[-1]}   <- compare this number across builds")

# Two independent failure modes, and the run must be judged on both:
#  1. LATE ARRIVAL - the cache is still growing after the residency window closed. That growth IS
#     the pop-in: those objects appeared because the camera looked at them.
#  2. INCOMPLETE RESIDENCY - the cache stops growing but at a number well below what the same scene
#     reaches with the window armed. Nothing "pops" because the geometry is simply never there.
#     A flatness-only test calls this a PASS, which is exactly wrong, so it is reported separately
#     and can only be judged against the other build's FINAL number.
tail_ent, tail_cap = ent[1:], cap[1:]
if not tail_ent:
    print("  INCONCLUSIVE: only one heartbeat - run longer (raise settle/duration)")
else:
    grew = max(tail_ent) - min(tail_ent)
    if grew == 0 and max(tail_cap) == 0.0:
        print(f"  NO LATE ARRIVAL: cache flat at {tail_ent[0]} with zero first-sight captures "
              f"after the window closed")
    else:
        print(f"  LATE ARRIVAL (pop-in): cache grew {grew} entries after the window, "
              f"peak cap={max(tail_cap)}/frame")
if not window:
    print("  !! no [postload-walk] line - the residency window never closed (unfixed build, or it "
          "was never armed). FINAL RESIDENT ENTRIES above is the number that matters.")
PY

echo
echo "[popin] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[popin] done"
