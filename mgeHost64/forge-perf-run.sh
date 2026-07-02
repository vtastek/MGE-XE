#!/usr/bin/env bash
# Forge automated perf harness: launch minimized (auto-loads test scene) -> poll host log for N new
# 'gpu split' heartbeats -> verify no device-removal -> kill both procs -> report the last N splits.
# Usage: forge-perf-run.sh [samples=5] [timeout=180]
set -u
SAMPLES="${1:-5}"
TIMEOUT="${2:-180}"
LOG="/mnt/c/mgem/morrowind64/mgeHost64.log"

startlines=0
[ -f "$LOG" ] && startlines=$(wc -l < "$LOG")
echo "[harness] start offset = $startlines lines; want $SAMPLES new 'gpu split' samples (timeout ${TIMEOUT}s)"

# Launch minimized (no focus steal). The save auto-loads.
powershell.exe -Command "Start-Process -FilePath 'Morrowind.exe' -WorkingDirectory 'C:\\mgem\\morrowind64' -WindowStyle Minimized" >/dev/null 2>&1
echo "[harness] launched Morrowind; polling..."

t0=$(date +%s)
got=0
while :; do
  sleep 3
  now=$(date +%s); el=$((now - t0))
  cur=0; [ -f "$LOG" ] && cur=$(wc -l < "$LOG")
  # The host TRUNCATES mgeHost64.log on launch → cur < startlines means the log rotated; measure from 0.
  if [ "$cur" -lt "$startlines" ]; then startlines=0; fi
  if [ "$cur" -gt "$startlines" ]; then
    got=$(tail -n +$((startlines + 1)) "$LOG" | grep -c "gpu split:")
  fi
  # crash guard: Morrowind gone with no samples
  running=$(powershell.exe -Command "(Get-Process Morrowind -ErrorAction SilentlyContinue) -ne \$null" 2>/dev/null | tr -d '\r\n ')
  echo "[harness] t=${el}s newlines=$((cur-startlines)) gpuSplits=$got running=$running"
  if [ "$got" -ge "$SAMPLES" ]; then echo "[harness] got $got samples"; break; fi
  if [ "$running" = "False" ] && [ "$got" -eq 0 ]; then echo "[harness] Morrowind EXITED with 0 samples (crash?)"; break; fi
  if [ "$el" -ge "$TIMEOUT" ]; then echo "[harness] TIMEOUT after ${el}s ($got samples)"; break; fi
done

echo "=== VERIFY (device removed / FAILED / fatal in new log) ==="
tail -n +$((startlines + 1)) "$LOG" | grep -Ei "device removed|FAILED|fatal|crash" | tail -20 || echo "  (clean)"

echo "=== last splits ==="
tail -n +$((startlines + 1)) "$LOG" | grep -E "host split:|gpu split:|\[dl\] exterior" | tail -$((SAMPLES * 3))

echo "[harness] killing procs..."
powershell.exe -Command "Stop-Process -Name Morrowind,mgeHost64 -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
echo "[harness] done"
