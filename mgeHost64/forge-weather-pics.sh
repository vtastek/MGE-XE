#!/usr/bin/env bash
# Ten weathers, ten pictures, unattended.
#
# WHY IT EXISTS: every look question about the sky is "what does it look like in <weather>", and the
# only rig that can answer it (forge-perf-run.sh) runs MINIMIZED with no keyboard behind it. Two
# pieces closed that: the weather ring's autoRun (MWSE parks a chosen weather after load) and the
# host's `dumpAtFrame` knob (arm the HDR/LDR dump at a frame number instead of numpad 1). This
# script is the two of them in a loop.
#
# Usage: forge-weather-pics.sh <outdir-tag> [save.ess] [hostKnobs] [weathers]
#   outdir-tag   a name for this arm, e.g. "baseline" or "mie3" -> hdrdump/pics/<tag>/
#   save.ess     default iboseydaneenwater.ess (shore + statics + water: where the defect shows)
#   hostKnobs    extra MGE_HOST_KNOBS, comma-separated; dumpAtFrame is added here
#   weathers     space-separated indices, default "0 1 2 3 4 5 6 7 8 9"
#
# ⚠ THE PICTURE IS NAMED BY WHAT THE HOST SAID WAS LIVE, NOT BY WHAT THE SCRIPT ASKED FOR. The
# `[hdrdump] ... weather=` field is parsed out of the run's own log and goes in the filename, and a
# mismatch against the requested index is printed loudly. A sweep that labels its output by intent
# is asserting something it never checked, and a mislabelled picture is worse than a missing one.
set -u
TAG="${1:-baseline}"
SAVE="${2:-iboseydaneenwater.ess}"
KNOBS_EXTRA="${3:-}"
WEATHERS="${4:-0 1 2 3 4 5 6 7 8 9}"

INSTALL="/mnt/c/mgem/morrowind64"
RINGCFG="$INSTALL/Data Files/MWSE/config/tw_weatherring.json"
DUMPDIR="$INSTALL/hdrdump"
OUT="$DUMPDIR/pics/$TAG"
LOG="$INSTALL/mgeHost64.log"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The dump frame. Frames tick only while the world renders, so this counts GAMEPLAY: ~2000 at ~73fps
# is ~27 s, comfortably past the ring's autoDelay + its 4 s inter-leg gap. SAMPLES must outlast it —
# the harness kills the run after N `gpu split` heartbeats and those are one per 300 frames.
DUMPFRAME=2000
SAMPLES=8
TIMEOUT=260

mkdir -p "$OUT"
RINGBAK="$(mktemp)"
cp "$RINGCFG" "$RINGBAK"
trap 'cp "$RINGBAK" "$RINGCFG"; rm -f "$RINGBAK"; echo "[pics] restored weather-ring config"' EXIT

for W in $WEATHERS; do
  echo "=============== weather $W ==============="
  python3 - "$RINGCFG" "$W" <<'PY'
import json, sys
path, w = sys.argv[1], int(sys.argv[2])
with open(path) as f: cfg = json.load(f)
cfg["autoRun"]     = True
cfg["mode"]        = "INSTANT"   # park it, do not walk it: a picture wants a SETTLED sky
cfg["autoControl"] = w           # leg 1 INSTANT -> this is what actually parks the weather
cfg["autoTarget"]  = w           # leg 2 then has nowhere to go, which is the point
cfg["autoDelay"]   = 6
cfg["enableOne"]   = True
with open(path, "w") as f: json.dump(cfg, f, indent=2)
PY

  KN="dumpAtFrame=$DUMPFRAME"
  [ -n "$KNOBS_EXTRA" ] && KN="$KN,$KNOBS_EXTRA"
  bash "$HERE/forge-perf-run.sh" "$SAMPLES" "$TIMEOUT" "$SAVE" "" "$KN" >/dev/null 2>&1

  # ⚠ THE WHOLE LOG, NOT A BYTE OFFSET TAKEN BEFORE THE RUN. mgeHost64.log is TRUNCATED at every
  # host start, so a `tail -c +N` saved beforehand points PAST the new end and returns nothing —
  # which is not "no dump", it is "the wrong question", and the first version of this script
  # reported every successful capture as a failure. One host start = one log, so the last matching
  # line in the file IS this run's. [[project_forge_scene_probe_truncates_log]]
  LINE=$(grep -a "\[hdrdump\]" "$LOG" 2>/dev/null | tail -1)
  if [ -z "$LINE" ]; then
    echo "[pics] weather $W: NO DUMP — no [hdrdump] line in the log (frame $DUMPFRAME never reached, or the dump DECLINED — it needs a scene-referred fp16+MSAA build)"
    continue
  fi
  SRC=$(printf '%s' "$LINE" | sed -n 's/.*hdrdump\\\(mge_[0-9]*\)\.exr.*/\1/p')
  GOT=$(printf '%s' "$LINE" | sed -n 's/.*weather=\([A-Za-z]*\)(\([0-9-]*\)).*/\1 \2/p')
  GOTNAME=$(printf '%s' "$GOT" | cut -d' ' -f1)
  GOTIDX=$(printf '%s' "$GOT" | cut -d' ' -f2)
  if [ "$GOTIDX" != "$W" ]; then
    echo "[pics] !! weather $W REQUESTED but host reports $GOTNAME($GOTIDX) — naming by what was LIVE"
  fi
  for EXT in tga exr; do
    [ -f "$DUMPDIR/$SRC.$EXT" ] && mv "$DUMPDIR/$SRC.$EXT" "$OUT/$(printf '%d' "${GOTIDX:-99}")_${GOTNAME:-unknown}.$EXT"
  done
  # ...and a PNG beside the TGA, because the point of the exercise is that somebody LOOKS at it and
  # a 32-bit TGA is not what a picture viewer opens by default. The TGA is kept: it is the exact
  # bytes the host read back, and the PNG is a convenience re-encode of it.
  python3 - "$OUT/${GOTIDX}_${GOTNAME}.tga" <<'PY' || echo "[pics] (PNG convert skipped)"
import sys
from PIL import Image
src = sys.argv[1]
Image.open(src).convert("RGB").save(src[:-4] + ".png")
PY
  echo "[pics] weather $W -> $OUT/${GOTIDX}_${GOTNAME}.png   ($LINE)"
  # The sky row that made this picture, kept beside it.
  grep -a "\[sky\] MEDIUM" "$LOG" | tail -1 >> "$OUT/${GOTIDX}_${GOTNAME}.txt"
  grep -a "\[atmos\]" "$LOG" | tail -1 >> "$OUT/${GOTIDX}_${GOTNAME}.txt"
  grep -a "apl-split" "$LOG" | tail -1 >> "$OUT/${GOTIDX}_${GOTNAME}.txt"
done
echo "[pics] done -> $OUT"
