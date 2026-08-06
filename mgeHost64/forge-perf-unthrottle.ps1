# Take Morrowind + mgeHost64 out of Windows' power-throttling (EcoQoS) bucket for a harness run.
#
# WHY THIS EXISTS. The harness launches minimized so it does not steal focus. Windows treats a
# minimized window as a background app and applies Power Throttling to it — and mgeHost64 has no
# window at all, so it is a permanent candidate. The measured effect on a laptop (mobile 4070,
# Balanced plan, on AC) was a near-uniform ~2.2x on EVERY number: GPU phases AND CPU record time
# alike, which is the signature of a clock change rather than a rendering change.
#
# That made absolute ms from a harness run incomparable to a real play session — a harness at
# 2048x1536 read ao=3.39 / host total=14.7 where the user's own foreground session on the same save
# and the same build read ao=1.51 / host total=6.1. A/B ratios measured within one run were still
# fine; every absolute number was not.
#
# ExecutionSpeed control bit set with a zero state = "do not throttle". High priority on top, since
# the same background classification costs scheduler priority too.
param([string[]]$Names = @('Morrowind', 'mgeHost64'))

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class MgePerfPower {
    [StructLayout(LayoutKind.Sequential)]
    public struct State { public uint Version; public uint ControlMask; public uint StateMask; }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetProcessInformation(IntPtr h, int infoClass, ref State info, uint size);

    // ProcessPowerThrottling = 4; PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1;
    // PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1. Control it, leave the state bit clear => off.
    public static bool Unthrottle(IntPtr h) {
        State s = new State();
        s.Version = 1; s.ControlMask = 1; s.StateMask = 0;
        return SetProcessInformation(h, 4, ref s, (uint)Marshal.SizeOf(typeof(State)));
    }
}
"@ -ErrorAction SilentlyContinue

foreach ($n in $Names) {
    $procs = @(Get-Process -Name $n -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) { Write-Output "[unthrottle] $n : not running"; continue }
    foreach ($p in $procs) {
        $ok = $false
        try { $ok = [MgePerfPower]::Unthrottle($p.Handle) } catch { $ok = $false }
        try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High } catch { }
        Write-Output ("[unthrottle] {0} pid={1} ecoqos-off={2} prio={3}" -f $n, $p.Id, $ok, $p.PriorityClass)
    }
}
