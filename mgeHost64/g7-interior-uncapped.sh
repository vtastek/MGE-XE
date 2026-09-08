#!/usr/bin/env bash
# Re-measure the INTERIOR row with our FPS cap lifted.
#
# WHY. `MGE.ini [Global Graphics] FPS Limit=163` is 6.135 ms, and our interior client frame floor
# measured 6.13 — i.e. the interior arm was reporting OUR OWN LIMITER, not the renderer, while the
# host's GPU floor underneath it sat at 4.74 ms. The fork ships no FPS-limit key at all, so its
# interior number is bound by real work. Comparing the two as-is compares a cap against a workload.
# Exteriors are unaffected (12-15 ms, nowhere near the cap); this is an interior-only correction.
#
# The cap is a CLIENT setting in mge3/MGE.ini, not a host knob, so it cannot be reached with
# MGE_HOST_KNOBS. It is edited here and restored on EXIT (including ctrl-C and crash), because
# leaving the user's frame cap raised after a measurement run would silently change normal play.
set -u
INI="/mnt/c/mgem/morrowind64/mge3/MGE.ini"
BAK="$(mktemp)"
cp "$INI" "$BAK"
trap 'cp "$BAK" "$INI"; rm -f "$BAK"; echo "[uncapped] restored MGE.ini FPS Limit"' EXIT

python3 - "$INI" <<'PY'
import sys, os, re
p = sys.argv[1]
# cp1252 + CRLF: MGE.ini is a Windows ini the GUI also writes. Read and write it in its own
# encoding, via temp + replace, so a failed encode cannot truncate the user's config.
with open(p, encoding="cp1252", newline="") as f:
    s = f.read()
new, n = re.subn(r"(?m)^FPS Limit=\d+\s*$", "FPS Limit=1000", s)
assert n == 1, "expected exactly one FPS Limit line, found %d" % n
tmp = p + ".tmp"
with open(tmp, "w", encoding="cp1252", newline="") as f:
    f.write(new)
os.replace(tmp, p)
print("[uncapped] FPS Limit -> 1000 (1.0 ms)")
PY

G7_ARMS=$'g7|mgeg7|\nours-def|morrowind64|\nours-lean|morrowind64|customResolve=0,mbEnable=0,objVelEnable=0\nours-allon-lean|morrowind64|reflWaterGate=1,reflGpuCull=1,reflHeightOcc=1,customResolve=0,mbEnable=0,objVelEnable=0' \
G7_SAVES=$'caius-interior|playerbalmoracaiuscosadeshouse.ess' \
  bash /mnt/c/projects/mgexe/MGE-XE/mgeHost64/g7-sweep.sh "${1:-2}" "${2:-2}" "${3:-300}" "${4:-/mnt/c/mgem/agentscratch/g7uncapped}"
