#!/usr/bin/env bash
# forge-perf-run.sh, but retried until it actually produces samples.
#
# WHY. `tes3.loadGame` intermittently aborts on the pinned baseline saves: it raises "Local count
# for script 'sleeperScript' (Patch for Purists.esm)' differs from local count for saved reference
# data" and returns to the main menu, so the host logs `[seam] render init ok ... sceneReady=0`,
# never receives a scene, and the run reports 0 `gpu split:` samples after the full timeout. On the
# next launch — same save, same env, same binaries — it loads fine. Roughly half the runs in one
# terrain-PBR A/B were lost this way.
#
# ⚠ IT IS NOT A DIALOG. Enumerating every top-level window of a stuck Morrowind.exe found no window
# of any dialog class, only the main one. So neither "spam space at it" nor a Win32 dismisser can
# help, and both were tried before the windows were actually looked at. The durable fix is a save
# whose script locals match the current plugin list, which has to be made in-game; until then, the
# cheap and honest mitigation is to notice the dead run and launch again.
#
# ⚠ A DEAD RUN MUST NEVER PASS AS DATA. The sample count is anchored to the heartbeat's own prefix,
# `[forge-hb] gpu split:`. An earlier version counted `grep -c "gpu split:"`, which also matches the
# harness's OWN status lines ("polling mgeHost64.log for 'gpu split:'"), so a run with zero real
# samples reported 2 — the guard against fake data was itself manufacturing it.
#
# ⚠ USE A SHORT PER-TRY TIMEOUT. The failure is intermittent (roughly half of launches on
# playerbalmora2.ess), not deterministic — two of the five runs in one A/B took it, and two
# consecutive retries took it again. A good load is in-world and emitting heartbeats inside ~45 s,
# so a dead run is identifiable long before a 200 s timeout expires; waiting the full timeout just
# multiplies the cost of every unlucky launch. 90 s per try with more tries is strictly better than
# 200 s with fewer: same coverage of the good case, a fifth of the wasted time on the bad one.
#
# Usage: forge-perf-retry.sh <outfile> <samples> <timeout> <save.ess> <renderScale> <hostKnobs> [tries]
set -u
OUTFILE="$1"; SAMPLES="$2"; TIMEOUT="$3"; SAVE="$4"; SCALE="$5"; KNOBS="$6"; TRIES="${7:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for try in $(seq 1 "$TRIES"); do
  bash "$HERE/forge-perf-run.sh" "$SAMPLES" "$TIMEOUT" "$SAVE" "$SCALE" "$KNOBS" > "$OUTFILE" 2>&1
  got=$(grep -c "forge-hb\] gpu split:" "$OUTFILE")
  if [ "$got" -gt 0 ]; then
    echo "[retry] try $try/$TRIES: $got samples"
    grep -oE "color=[0-9.]+" "$OUTFILE" | sed 's/color=//' | tr '\n' ' '; echo
    exit 0
  fi
  echo "[retry] try $try/$TRIES: DEAD RUN (0 samples — load aborted, see MWSE.log for the script-local warning)"
  # Make sure nothing is left holding the seam before relaunching: a relaunch inside the previous
  # process's teardown is its own failure mode, separate from the load abort.
  powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
  for _ in $(seq 1 30); do
    n=$(powershell.exe -Command "(Get-Process -Name Morrowind,mgeHost64 -ErrorAction SilentlyContinue | Measure-Object).Count" 2>/dev/null | tr -d '\r')
    [ "${n:-0}" = "0" ] && break
    sleep 2
  done
  sleep 10
done
echo "[retry] GAVE UP after $TRIES tries — no samples. Do NOT read a number out of this."
exit 1
