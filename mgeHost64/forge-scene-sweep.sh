#!/usr/bin/env bash
# Forge scene sweep: run forge-perf-run.sh across the FIXED set of baseline saves, keep each run's
# logs, and print one comparison table.
#
# WHY THIS EXISTS. Single-scene numbers do not generalise and we have been burned by treating one as
# if it did: the shadow-mask cost (1.65-2.53ms, ~49% of host GPU) was measured in a light-dense
# interior and got written down as "the biggest GPU item" full stop — in an exterior the same
# dispatch is 0.23ms. The cost model is O(pixels x per-pixel covering light slots), so the scene
# CLASS is the independent variable and any perf claim without one attached is unfalsifiable.
#
# The two floors (justsky / justwall) are the point of the set: they give the fixed per-frame cost,
# so every other scene reads as a delta over the floor rather than an absolute nobody can interpret.
#
# Usage: forge-scene-sweep.sh [samples=4] [timeout=180]
set -u
SAMPLES="${1:-4}"
TIMEOUT="${2:-180}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${SWEEP_OUT:-/tmp/forge-sweep}"
HOSTLOG="/mnt/c/mgem/morrowind64/mgeHost64.log"
CLIENTLOG="/mnt/c/mgem/morrowind64/mgeXE.log"

# Ordered floors-first so the table reads bottom-up from the fixed cost.
SAVES=(
  vjustsky.ess
  vjustwall.ess
  vlightdense.ess
  vmanylights.ess
  vdensecity.ess
  vheavydistantland.ess
)

mkdir -p "$OUT"
echo "[sweep] $((${#SAVES[@]})) scenes x $SAMPLES samples -> $OUT"

for save in "${SAVES[@]}"; do
  tag="${save%.ess}"
  echo ""
  echo "=================== $tag ==================="
  bash "$HERE/forge-perf-run.sh" "$SAMPLES" "$TIMEOUT" "$save" > "$OUT/$tag.run.txt" 2>&1
  tail -3 "$OUT/$tag.run.txt"
  # Only keep the logs if the run actually produced samples. On a failed launch the host log still
  # holds the PREVIOUS scene's content, and copying it would publish a confident, wrong row.
  if grep -q "got [0-9]* samples" "$OUT/$tag.run.txt"; then
    cp "$HOSTLOG"   "$OUT/$tag.host.log"   2>/dev/null || echo "[sweep] no host log for $tag"
    cp "$CLIENTLOG" "$OUT/$tag.client.log" 2>/dev/null || true
  else
    echo "[sweep] !! $tag produced no samples — row will be blank, NOT stale"
  fi
  sleep 4
done

echo ""
python3 "$HERE/forge-sweep-report.py" "$OUT"
