# Catch the STUCK state and enumerate every window in it.
#
# !! WHY THIS EXISTS, AND THE MISTAKE IT FIXES. A first probe enumerated Morrowind's windows 45 s
# after launch, found nothing but the main window, and concluded "there is no dialog, so neither
# space-spam nor a Win32 dismisser can help". That conclusion was drawn from the WRONG STATE: the
# game had already loaded by the time the probe ran -- the user was pressing space to rescue runs
# while the tests were going. Enumerating a rescued process says nothing about a stuck one.
#
# So this probe does not sleep for a fixed time and look once. It watches for the stuck SIGNATURE
# and enumerates at that moment:
#   * MWSE.log gains "in function 'loadGame'"  (the script-local warning fired), or
#   * the host log has `sceneReady=0` and has stopped growing for several seconds.
# Then it dumps every top-level window and every child, with class, title and visibility.
#
# Run it BEFORE launching the game, in parallel with the harness, and do NOT press space during it --
# a rescued launch is exactly the sample that cannot answer the question.
param(
    [string]$Install = "C:\mgem\morrowind64",
    [int]$Seconds = 150
)

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public class StuckProbe {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr parent, EnumProc cb, IntPtr p);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder b, int m);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder b, int m);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern bool IsWindowEnabled(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr GetWindow(IntPtr h, uint cmd);
    public static string Cls(IntPtr h) { var s = new StringBuilder(128); GetClassName(h, s, s.Capacity); return s.ToString(); }
    public static string Txt(IntPtr h) { var s = new StringBuilder(512); GetWindowText(h, s, s.Capacity); return s.ToString(); }
    public static System.Collections.Generic.List<IntPtr> Top(uint pid) {
        var l = new System.Collections.Generic.List<IntPtr>();
        EnumWindows((h,p) => { uint w; GetWindowThreadProcessId(h, out w); if (w==pid) l.Add(h); return true; }, IntPtr.Zero);
        return l;
    }
    public static System.Collections.Generic.List<IntPtr> Kids(IntPtr parent) {
        var l = new System.Collections.Generic.List<IntPtr>();
        EnumChildWindows(parent, (h,p) => { l.Add(h); return true; }, IntPtr.Zero);
        return l;
    }
}
"@

$mwse = Join-Path $Install "MWSE.log"
$hostlog = Join-Path $Install "mgeHost64.log"
$deadline = (Get-Date).AddSeconds($Seconds)
$lastHostSize = -1
$stalledTicks = 0

function Dump-Windows($why) {
    Write-Output "================ STUCK STATE DETECTED: $why ================"
    Write-Output ("time {0}" -f (Get-Date -Format "HH:mm:ss"))
    $p = Get-Process -Name Morrowind -ErrorAction SilentlyContinue
    if (-not $p) { Write-Output "  (Morrowind not running)"; return }
    foreach ($proc in $p) {
        Write-Output "=== pid $($proc.Id)  responding=$($proc.Responding)  mainHwnd=$($proc.MainWindowHandle) ==="
        foreach ($h in [StuckProbe]::Top([uint32]$proc.Id)) {
            Write-Output ("  TOP hwnd={0} vis={1} enabled={2} class='{3}' title='{4}'" -f `
                $h, [StuckProbe]::IsWindowVisible($h), [StuckProbe]::IsWindowEnabled($h), `
                [StuckProbe]::Cls($h), [StuckProbe]::Txt($h))
            foreach ($c in [StuckProbe]::Kids($h)) {
                $ct = [StuckProbe]::Txt($c)
                Write-Output ("    kid hwnd={0} vis={1} enabled={2} class='{3}' title='{4}'" -f `
                    $c, [StuckProbe]::IsWindowVisible($c), [StuckProbe]::IsWindowEnabled($c), `
                    [StuckProbe]::Cls($c), $ct)
            }
        }
    }
    Write-Output "============================================================"
}

Write-Output "[probe] watching for the stuck signature (do NOT press space during this run)"
while ((Get-Date) -lt $deadline) {
    if (Test-Path $mwse) {
        $hit = Select-String -Path $mwse -Pattern "in function 'loadGame'" -SimpleMatch -ErrorAction SilentlyContinue
        if ($hit) {
            Dump-Windows "MWSE.log shows the loadGame warning"
            exit 0
        }
    }
    if (Test-Path $hostlog) {
        $sz = (Get-Item $hostlog).Length
        $ready = Select-String -Path $hostlog -Pattern "sceneReady=0" -SimpleMatch -ErrorAction SilentlyContinue
        if ($ready) {
            if ($sz -eq $lastHostSize) { $stalledTicks++ } else { $stalledTicks = 0 }
            $lastHostSize = $sz
            if ($stalledTicks -ge 12) {   # ~12 s of no growth past seam init
                Dump-Windows "host log frozen at sceneReady=0 for ~12s"
                exit 0
            }
        }
    }
    Start-Sleep -Milliseconds 1000
}
Write-Output "[probe] no stuck state seen within ${Seconds}s -- the run loaded normally"
