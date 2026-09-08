#!/usr/bin/env bash
# mgeg7 comparison sweep: three saves x three arms, INTERLEAVED across installs.
#
# WHY INTERLEAVED, and not all-of-one-then-all-of-the-other. A machine drifts over a sweep - thermals,
# EcoQoS, whatever else Windows decides to do - and the last recorded case of running one arm to
# completion before the other produced a REVERTED arm that read worse than the change it was meant to
# refute (feedback_perf_ab_must_be_interleaved). Rounds alternate installs so any drift lands on both
# arms equally, and a per-round comparison stays valid even if the absolute numbers wander.
#
# WHY THE WEATHER IS RECORDED PER ROW. Loading a save re-rolls the weather (project_forge_apl_weather_roll),
# so two rows of the same save can be an overcast frame against an ashstorm frame. The fpsprobe line
# carries the weather it was measured under; rows whose weather does not match are a pair to discard,
# not a result. Nothing here can force the weather - the discarding is done when reading the table.
#
# Every run's full harness output is kept, because the summary line below is a grep and a grep that
# silently matched nothing looks exactly like a scene that got faster.
set -u
ROUNDS="${1:-2}"
SAMPLES="${2:-3}"
TIMEOUT="${3:-300}"
OUT="${4:-/mnt/c/mgem/agentscratch/g7sweep}"
HARNESS="/mnt/c/projects/mgexe/MGE-XE/mgeHost64/forge-perf-run.sh"

mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

# arm := label|install|knobs. 'ours-allon' is the everything-on arm the user asked to be measured
# alongside shipped defaults; reflHeightOcc does nothing without reflGpuCull, so both are set.
# Overridable so a follow-up question does not need a second copy of this script. G7_ARMS/G7_SAVES
# take the same "label|install|knobs" / "label|save.ess" form, newline-separated.
if [ -n "${G7_ARMS:-}" ]; then
  IFS=$'\n' read -r -d '' -a ARMS <<< "$G7_ARMS" || true
else
ARMS=(
  "g7|mgeg7|"
  "ours-def|morrowind64|"
  "ours-allon|morrowind64|reflWaterGate=1,reflGpuCull=1,reflHeightOcc=1"
)
fi
if [ -n "${G7_SAVES:-}" ]; then
  IFS=$'\n' read -r -d '' -a SAVES <<< "$G7_SAVES" || true
else
SAVES=(
  "ascadian-nowater|playerascadianislesregionnowater.ess"
  "balmora23|playerbalmora23.ess"
  "caius-interior|playerbalmoracaiuscosadeshouse.ess"
)
fi

for r in $(seq 1 "$ROUNDS"); do
  for sv in "${SAVES[@]}"; do
    svname="${sv%%|*}"; svfile="${sv##*|}"
    for arm in "${ARMS[@]}"; do
      IFS='|' read -r label install knobs <<< "$arm"
      tag="r${r}-${svname}-${label}"
      echo "=== $tag ==="
      MGE_INSTALL="$install" MGE_METRIC=fpsprobe \
        bash "$HARNESS" "$SAMPLES" "$TIMEOUT" "$svfile" "" "$knobs" > "$OUT/$tag.log" 2>&1
      # The fpsprobe rows, tagged with the arm they came from. The host floor is appended for our
      # arms only - the fork writes no such line, and an empty section there is expected, not a fault.
      grep -E "^\[fpsprobe\] n=" "$OUT/$tag.log" | sed "s/^/[$tag] /" | tee -a "$SUMMARY"
      grep -E "gpu stall latch:" "$OUT/$tag.log" | sed -E "s/.*frame min=([0-9.]+) MEAN=([0-9.]+).*/[$tag] HOSTFLOOR min=\1 mean=\2/" | tee -a "$SUMMARY"
    done
  done
done

echo
echo "=== SUMMARY ($SUMMARY) ==="
cat "$SUMMARY"
