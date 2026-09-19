# Dismiss Morrowind's NATIVE warning dialogs, for the perf harness.
#
# WHY. The pinned baseline saves drift against the plugin list, and `tes3.loadGame` then raises
# Morrowind's own warning box from inside the load:
#
#   Morrowind has raised a warning with a lua stack trace: Local count for script 'sleeperScript'
#   (Patch for Purists.esm)' differs from local count for saved reference data.
#       [C]: in function 'loadGame'
#
# It is MODAL on the main thread, so the game freezes mid-load: the host logs
# `[seam] render init ok ... sceneReady=0`, never receives a scene, never prints a `gpu split:`
# heartbeat, and the run times out with 0 samples. In the log that is indistinguishable from a host
# hang, which is how several terrain-PBR A/B runs were lost to a hunt for a rendering bug that was
# not there.
#
# !! THIS IS A NATIVE DIALOG, NOT AN IN-ENGINE MENU, and the difference decides the whole approach.
# The MWSE-side companion (mgeHost64/autodismiss) hooks `uiActivated` on `MenuMessage` and cannot
# see this one at all -- it is a Win32 window of class #32770 owned by Morrowind.exe, put up before
# any tes3ui menu exists. So it is dismissed from OUTSIDE, by posting to the dialog itself.
#
# Posting, not typing. `SendKeys` would need the foreground window, and stealing focus is the one
# thing this harness exists not to do ([[feedback_dont_drive_the_game]]) -- a measurement that
# requires the foreground is one nobody can run while working. A native dialog honours a posted
# WM_COMMAND/IDOK without ever being focused, which is why "spam space at the game window" does not
# work while this does: Morrowind reads the keyboard through DirectInput, which never sees a posted
# WM_KEYDOWN, but the DIALOG is an ordinary Win32 control that does.
param(
    [int]$Seconds = 120,
    [string]$ProcName = "Morrowind"
)

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public class MgeDlg {
    public delegate bool EnumProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder buf, int max);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder buf, int max);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr w, IntPtr l);

    public const uint WM_COMMAND = 0x0111;
    public const int  IDOK       = 1;
    public const int  IDYES      = 6;

    // Every visible #32770 (the standard dialog class) owned by `pid`.
    public static System.Collections.Generic.List<IntPtr> Dialogs(uint pid) {
        var hits = new System.Collections.Generic.List<IntPtr>();
        EnumWindows((h, p) => {
            uint wpid; GetWindowThreadProcessId(h, out wpid);
            if (wpid != pid) { return true; }
            if (!IsWindowVisible(h)) { return true; }
            var cn = new StringBuilder(64);
            GetClassName(h, cn, cn.Capacity);
            if (cn.ToString() == "#32770") { hits.Add(h); }
            return true;
        }, IntPtr.Zero);
        return hits;
    }

    public static string Title(IntPtr h) {
        var sb = new StringBuilder(512);
        GetWindowText(h, sb, sb.Capacity);
        return sb.ToString();
    }
}
"@

$deadline = (Get-Date).AddSeconds($Seconds)
$seen = @{}
$count = 0

while ((Get-Date) -lt $deadline) {
    $proc = Get-Process -Name $ProcName -ErrorAction SilentlyContinue
    if (-not $proc) {
        # Not up yet, or already gone. Either way there is nothing to dismiss this tick.
        Start-Sleep -Milliseconds 400
        continue
    }
    foreach ($p in $proc) {
        foreach ($h in [MgeDlg]::Dialogs([uint32]$p.Id)) {
            $title = [MgeDlg]::Title($h)
            if (-not $seen.ContainsKey($h)) {
                $seen[$h] = $true
                Write-Output "[dismiss] dialog '$title' (hwnd $h) -- posting IDOK"
            }
            # IDOK covers the OK-only warnings; IDYES covers the confirm variants. Posting a control
            # id the dialog does not have is simply ignored, so both are safe to send.
            [void][MgeDlg]::PostMessage($h, [MgeDlg]::WM_COMMAND, [IntPtr][MgeDlg]::IDOK,  [IntPtr]::Zero)
            [void][MgeDlg]::PostMessage($h, [MgeDlg]::WM_COMMAND, [IntPtr][MgeDlg]::IDYES, [IntPtr]::Zero)
            $count++
        }
    }
    Start-Sleep -Milliseconds 400
}
Write-Output "[dismiss] done ($count posts to $($seen.Count) distinct dialogs)"
