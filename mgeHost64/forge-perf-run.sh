#!/usr/bin/env bash
# Forge automated perf harness: launch minimized (auto-loads test scene) -> poll host log for N new
# 'gpu split' heartbeats -> verify no device-removal -> kill both procs -> report the last N splits.
# Usage: forge-perf-run.sh [samples=5] [timeout=180] [save.ess] [renderScale] [hostKnobs] [trace] [clientEnv]
#
# clientEnv (7th arg) = comma-separated NAME=VALUE pairs set in the game's environment, for client
# arms that are env-driven (e.g. MGE_TIER1_SEM=0 forces the event handoff on Windows). The names in
# CLIENT_ENV_NAMES below are always cleared first, so a stale user variable cannot label the arm.
#
# trace (6th arg, any non-empty value) = MGE_FRAME_TRACE=1 for both processes, without an env-var
# prefix on the command line (a prefix makes Claude Code's permission check prompt every run).
# Empty strings skip the args before it: forge-perf-run.sh 8 300 vseydaneen.ess "" "" trace
#
# hostKnobs (5th arg) is passed through as MGE_HOST_KNOBS="name=value,name=value" and applied by the
# host at startup (ForgeRender::applyEnvOverrides). It exists because every look knob is a dev-panel
# widget and this harness runs MINIMIZED on purpose — so without it, any A/B that turns on a
# checkbox simply cannot be measured here, which is how a brightness complaint and a physical
# derivation argued past each other for two builds with no shared number. The host LOGS what it
# applied, so each run's log carries the arm it was measured in.
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
# WHICH INSTALL. Defaults to the dev deploy target; MGE_INSTALL names another directory under
# C:\mgem (e.g. MGE_INSTALL=mwdlss). Added 2026-09-02 because a ~90ms/frame regression reproduced
# ONLY in the mwdlss install — the one this harness could not point at — so the single environment
# that showed the bug was the single one with no way to measure it.
INSTALL="${MGE_INSTALL:-morrowind64}"
DIR="/mnt/c/mgem/$INSTALL"
WINDIR="C:\\mgem\\$INSTALL"
if [ ! -d "$DIR" ]; then
  echo "[harness] ERROR: no such install: $DIR" >&2
  exit 1
fi
echo "[harness] install = $INSTALL"
SAMPLES="${1:-5}"
TIMEOUT="${2:-180}"
SAVE="${3:-}"
SCALE="${4:-}"
KNOBS="${5:-}"
if [ -n "${6:-}" ]; then MGE_FRAME_TRACE=1; fi
CLIENTENV="${7:-}"
CLIENT_ENV_NAMES="MGE_TIER1_SEM MGE_TIER1_EVENT MGE_COPY_AT_BLIT MGE_FRAME_AHEAD MGE_GEOM_ALIAS MGE_FREEZE_CLOCK MGE_TEX_STREAM_ASYNC MGE_TEX_IO_THREAD"
LOG="$DIR/mgeHost64.log"
CFG="$DIR/Data Files/MWSE/config/instant load.json"
CFGBAK="$(mktemp)"

# WHICH METRIC. Default 'gpusplit' is this script's original behaviour, unchanged: poll the host log
# for 'gpu split:' heartbeats and report the host-side sections. 'fpsprobe' polls MWSE.log for the
# [fpsprobe] mod's lines instead, and skips every host-only section.
#
# It exists because the mgeg7 comparison install (Greatness7's DX9 fork) has no 'gpu split:' line and
# no equivalent of one - and its own numbers have no counterpart here either. The only quantity both
# builds can be asked for with the SAME instrument is the client frame time, which the fpsprobe MWSE
# mod reports from enterFrame.delta in whichever install it is deployed to. One script, one metric,
# both installs; the alternative is two rulers and a table nobody can defend.
#
# NOTE the fork ships its OWN mgeHost64.exe writing its OWN mgeHost64.log. Same names as ours. The
# kill/archive paths below therefore work unchanged for it, but never run both installs at once.
METRIC="${MGE_METRIC:-gpusplit}"
case "$METRIC" in
  gpusplit) MLOG="$LOG"          ; MPAT="gpu split:" ; MLABEL="gpuSplits" ;;
  # `n=` anchors the pattern to a REPORTED WINDOW. The mod also prints two banner lines per load
  # ("loaded - settling", "sampling STARTED"), and matching bare [fpsprobe] counted those as samples:
  # a run asked for 3 got one 600-frame window and stopped, with nothing in the output saying it had
  # measured a third of what was requested.
  fpsprobe) MLOG="$DIR/MWSE.log" ; MPAT="\[fpsprobe\] n=" ; MLABEL="fpsWindows" ;;
  *) echo "[harness] ERROR: unknown MGE_METRIC=$METRIC (want gpusplit or fpsprobe)" >&2; exit 1 ;;
esac
echo "[harness] metric = $METRIC (polling $(basename "$MLOG") for '$MPAT')"

if [ -n "$SAVE" ]; then
  if [ ! -f "$DIR/Saves/$SAVE" ]; then
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
ARCHIVE="$DIR/logarchive"
mkdir -p "$ARCHIVE"
stamp=$(date +%Y%m%d-%H%M%S)
for f in "$LOG" "$DIR/mgeXE.log" "$DIR/MWSE.log"; do
  [ -s "$f" ] && cp "$f" "$ARCHIVE/$(basename "$f" .log)-$stamp.log" 2>/dev/null
done
# Keep the 20 most recent of each; these run to tens of MB.
ls -1t "$ARCHIVE"/mgeHost64-*.log 2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/mgeXE-*.log     2>/dev/null | tail -n +21 | xargs -r rm -f
ls -1t "$ARCHIVE"/MWSE-*.log      2>/dev/null | tail -n +21 | xargs -r rm -f

startlines=0
[ -f "$LOG" ] && startlines=$(wc -l < "$LOG")
# The METRIC log's own offset. In gpusplit mode this is the same file and the same number, so the
# poll loop below behaves identically; in fpsprobe mode it is MWSE.log, which MWSE truncates at
# startup exactly as the host truncates its own, so the same rotation reset covers both.
mstartlines=0
[ -f "$MLOG" ] && mstartlines=$(wc -l < "$MLOG")
# Same offset trick on the CLIENT log. The host's numbers alone cannot answer "is the host the
# bottleneck" — only the client's render=[host=] / overlap= pair says how much of the host frame the
# client actually waited for. mgecore does NOT truncate mgeXE.log, so this offset is what separates
# this run from the session before it.
CLOG="$DIR/mgeXE.log"
cstartlines=0
[ -f "$CLOG" ] && cstartlines=$(wc -l < "$CLOG")
echo "[harness] start offset = $mstartlines lines; want $SAMPLES new '$MPAT' samples (timeout ${TIMEOUT}s)"

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
# Built as ONE prefix rather than a branch per option: with two independent env vars the branchy
# form needs four arms, and the arm nobody exercises is the one that silently drops a variable.
ENVSET="$RDOC_STRIP"
if [ -n "$SCALE" ]; then
  echo "[harness] render scale = ${SCALE}x (MGE_RENDER_SCALE)"
  ENVSET="${ENVSET}\$env:MGE_RENDER_SCALE='$SCALE'; "
fi
# AUTO-DISMISS the startup confirmation dialogs (mgeHost64/autodismiss -> mods/mgexe/autodismiss).
# The pinned baseline saves predate the current plugin list, so tes3.loadGame raises a "content
# files have changed" MenuMessage and, with nobody at the keyboard of a MINIMIZED window, the run
# sits at it until timeout: the host logs `sceneReady=0`, never gets a scene, and reports 0 samples.
# That is indistinguishable in the log from a host hang, which is how several terrain-PBR A/B runs
# were lost to a hunt for a rendering bug that did not exist.
#
# Keystrokes cannot do this: Morrowind reads the keyboard through DirectInput, which never sees a
# PostMessage'd WM_KEYDOWN, so dismissing it from outside would mean stealing focus — the one thing
# this harness exists to avoid. The mod presses the button through MWSE instead, and disarms itself
# the moment the save is loaded so it can never reach an in-game dialog.
ENVSET="${ENVSET}\$env:MGE_AUTODISMISS='1'; "

# ExactPos A/B arm (client, src/mge/exactpos.h), taken from the caller's env. ALWAYS written for the
# same reason as MGE_HOST_KNOBS below: a stale value in the user environment would mislabel the arm.
if [ -n "${MGE_EXACT_POS:-}" ]; then
  echo "[harness] MGE_EXACT_POS=$MGE_EXACT_POS"
  ENVSET="${ENVSET}\$env:MGE_EXACT_POS='$MGE_EXACT_POS'; "
else
  ENVSET="${ENVSET}Remove-Item Env:MGE_EXACT_POS -ErrorAction SilentlyContinue; "
fi

# Frame timeline trace (src/ipc/frametrace.h), both processes: MGE_FRAME_TRACE=1 from the caller's env.
# Dumps land beside Morrowind.exe; render them with mgexe-devkit/tools/frametrace-{summary,html}.py.
if [ -n "${MGE_FRAME_TRACE:-}" ]; then
  echo "[harness] MGE_FRAME_TRACE=$MGE_FRAME_TRACE"
  ENVSET="${ENVSET}\$env:MGE_FRAME_TRACE='$MGE_FRAME_TRACE'; "
else
  ENVSET="${ENVSET}Remove-Item Env:MGE_FRAME_TRACE -ErrorAction SilentlyContinue; "
fi

# ALWAYS written, even when empty — a stale MGE_HOST_KNOBS left in the user environment would ride
# along in every run exactly the way MGE_RDOC did for three days, and the arm would be mislabelled.
if [ -n "$KNOBS" ]; then
  echo "[harness] host knobs = $KNOBS (MGE_HOST_KNOBS)"
  ENVSET="${ENVSET}\$env:MGE_HOST_KNOBS='$KNOBS'; "
else
  ENVSET="${ENVSET}Remove-Item Env:MGE_HOST_KNOBS -ErrorAction SilentlyContinue; "
fi
for n in $CLIENT_ENV_NAMES; do
  ENVSET="${ENVSET}Remove-Item Env:$n -ErrorAction SilentlyContinue; "
done
if [ -n "$CLIENTENV" ]; then
  echo "[harness] client env = $CLIENTENV"
  IFS=',' read -r -a _pairs <<< "$CLIENTENV"
  for p in "${_pairs[@]}"; do
    ENVSET="${ENVSET}\$env:${p%%=*}='${p#*=}'; "
  done
fi
powershell.exe -Command "${ENVSET}Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory '$WINDIR' -WindowStyle Minimized" >/dev/null 2>&1

# A watcher for Morrowind's NATIVE warning boxes (Win32 #32770), kept as cheap insurance.
#
# ⚠ IT IS NOT THE FIX FOR THE 0-SAMPLE RUNS, AND THE DIALOG THEORY THAT BUILT IT WAS WRONG.
# Enumerating EVERY top-level window of a stuck Morrowind.exe found no dialog of any class — only
# the main 'Morrowind' window plus invisible IME/d3d helpers. So there is nothing to dismiss and
# nothing a keystroke could have reached either. What actually happens: `tes3.loadGame` sometimes
# raises "Local count for script 'sleeperScript' (Patch for Purists.esm)' differs from local count
# for saved reference data" and ABORTS the load, leaving the game at the main menu; on other
# launches, same save and same env, the warning does not fire and the load succeeds. It is a race,
# not a modal box — which is why the retry below is the real mitigation.
DISMISS_LOG="$(mktemp)"
powershell.exe -ExecutionPolicy Bypass -File "$(wslpath -w "$(dirname "$0")/dismiss-dialogs.ps1")" \
  -Seconds "$((TIMEOUT + 20))" > "$DISMISS_LOG" 2>&1 &
DISMISS_PID=$!
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
    unthrottle_on=mgeHost64
    [ "$METRIC" = "fpsprobe" ] && unthrottle_on=Morrowind
    hostalive=$(powershell.exe -Command "@(Get-Process $unthrottle_on -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
    if [ "${hostalive:-0}" -gt 0 ]; then
      powershell.exe -ExecutionPolicy Bypass -File 'C:\projects\mgexe\MGE-XE\mgeHost64\forge-perf-unthrottle.ps1' 2>&1 | sed 's/^/[harness] /'
      unthrottled=1
    fi
  fi
  now=$(date +%s); el=$((now - t0))
  cur=0; [ -f "$LOG" ] && cur=$(wc -l < "$LOG")
  # The host TRUNCATES mgeHost64.log on launch → cur < startlines means the log rotated; measure from 0.
  if [ "$cur" -lt "$startlines" ]; then startlines=0; fi
  mcur=0; [ -f "$MLOG" ] && mcur=$(wc -l < "$MLOG")
  if [ "$mcur" -lt "$mstartlines" ]; then mstartlines=0; fi
  if [ "$mcur" -gt "$mstartlines" ]; then
    got=$(tail -n +$((mstartlines + 1)) "$MLOG" | grep -c "$MPAT")
  fi
  # crash guard: Morrowind gone with no samples
  # @(...).Count, not "-ne $null": with two Morrowind handles alive the latter formats BOTH process
  # objects into the output and the comparison silently becomes garbage.
  nproc=$(powershell.exe -Command "@(Get-Process Morrowind -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r\n ')
  running=$([ "${nproc:-0}" -gt 0 ] && echo True || echo False)
  echo "[harness] t=${el}s newlines=$((mcur-mstartlines)) ${MLABEL}=$got running=$running"
  if [ "$got" -ge "$SAMPLES" ]; then echo "[harness] got $got samples"; break; fi
  if [ "$running" = "False" ] && [ "$got" -eq 0 ]; then echo "[harness] Morrowind EXITED with 0 samples (crash?)"; break; fi
  if [ "$el" -ge "$TIMEOUT" ]; then echo "[harness] TIMEOUT after ${el}s ($got samples)"; break; fi
done

if [ "$METRIC" = "fpsprobe" ]; then

echo "=== VERIFY (errors in new MWSE.log) ==="
tail -n +$((mstartlines + 1)) "$MLOG" | grep -Ei "device removed|FAILED|fatal|crash|lua error" | tail -20 || echo "  (clean)"

# min is the headline. See the fpsprobe mod's header: a median over a window measures how many
# stalled frames the window caught, not what the renderer costs.
echo "=== fpsprobe (min is the headline) ==="
tail -n +$((mstartlines + 1)) "$MLOG" | grep -E "\[fpsprobe\]" | tail -$((SAMPLES + 2))

# Printed for OUR arms only, as supporting detail; the fork writes neither line. Absent output here
# is expected in the G7 arm and is not a failed run.
echo "=== host gpu floor (ours only; 'frame min' is the floor) ==="
tail -n +$((startlines + 1)) "$LOG" 2>/dev/null | grep -E "gpu stall latch:" | tail -3 || echo "  (none)"
echo "=== client (mgeXE.log; ours only) ==="
tail -n +$((cstartlines + 1)) "$CLOG" 2>/dev/null \
  | grep -E "\[seam\] backbuffer|\[hb\] [0-9]+ frames avg:" | tail -6 || echo "  (none)"

else

echo "=== VERIFY (device removed / FAILED / fatal in new log) ==="
tail -n +$((startlines + 1)) "$LOG" | grep -Ei "device removed|FAILED|fatal|crash" | tail -20 || echo "  (clean)"

echo "=== last splits ==="
tail -n +$((startlines + 1)) "$LOG" | grep -E "host split:|gpu split:|gpu color sub:|\[dl\] exterior|dist lights " | tail -$((SAMPLES * 4))

# The METERING lines, reported by the harness itself rather than left to a later grep. The knob arm
# a run was taken in is only recoverable from the log, so the arm and its numbers belong in the same
# captured output — the alternative is a table of readings whose labels come from memory.
echo "=== knobs applied ==="
tail -n +$((startlines + 1)) "$LOG" | grep -E "MGE_HOST_KNOBS|UNKNOWN knob" | tail -8 || echo "  (none — default build)"
# ...AND WHAT THE RUN IS ACTUALLY AT. Since 2026-09-21 the dev panel can SAVE its knobs to
# mgeHostPanel.ini and the host loads that at startup, so "I passed no knobs" no longer implies "this
# is the build's baseline". MGE_HOST_KNOBS is re-applied over the file and still wins, but anything
# the arm does not name comes from whatever was last saved in a play session. This line is the whole
# answer in one number, and it is here rather than left in the log because a contaminated baseline
# that nobody looked for is exactly how a table of readings goes quietly wrong.
tail -n +$((startlines + 1)) "$LOG" | grep -E "\[panel\] .* off the build default|\[panel\] mgeHostPanel" | tail -3
echo "=== apl / apl-split ==="
tail -n +$((startlines + 1)) "$LOG" | grep -E "\[forge-hb\] apl" | tail -6

# The client side of the same frames. [seam] backbuffer is printed FIRST so every table row carries
# the resolution it was actually measured at, rather than the one that was asked for.
echo "=== client (mgeXE.log) ==="
if [ "$(wc -l < "$CLOG" 2>/dev/null || echo 0)" -lt "$cstartlines" ]; then cstartlines=0; fi
tail -n +$((cstartlines + 1)) "$CLOG" 2>/dev/null \
  | grep -E "\[seam\] backbuffer|MGE_RENDER_SCALE|\[hb\] [0-9]+ frames avg:|\[hb\] host recv:|\[produce\] overlap:" \
  | tail -12 || echo "  (no client heartbeats)"

fi

if [ -n "${DISMISS_PID:-}" ]; then
  kill "$DISMISS_PID" 2>/dev/null
  wait "$DISMISS_PID" 2>/dev/null
fi
if [ -s "${DISMISS_LOG:-/dev/null}" ] && grep -q "posting IDOK" "${DISMISS_LOG:-/dev/null}" 2>/dev/null; then
  echo "=== dialogs dismissed (this run needed rescuing — the save has drifted) ==="
  grep "posting IDOK" "$DISMISS_LOG" | head -6
fi
rm -f "${DISMISS_LOG:-}" 2>/dev/null

echo "[harness] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[harness] done"
