# Dismiss modal dialogs belonging to Morrowind during a MINIMIZED harness run.
#
# WHY. The harness launches minimized on purpose (no focus steal) and nobody is at the keyboard by
# design. A modal warning dialog therefore blocks the game forever, and the failure is
# INDISTINGUISHABLE from a hang or a code fault: mgeHost64.log stops right after "Distant
# worldspaces memory use", the client log shows `ui=61` with every world counter 0, and the harness
# reports "TIMEOUT (0 samples)". That cost two 12-minute runs in one session before it was named.
#
# ⚠ WHY NOT SendKeys. WScript.Shell SendKeys requires the target window to be FOREGROUND, so it
# would both steal focus (the thing the minimized launch exists to avoid) and fail whenever the user
# is typing elsewhere. PostMessage is delivered to a specific window handle and needs no focus at
# all, so this stays invisible to whoever is using the machine.
#
# ⚠ SCOPED TO ONE PROCESS. It only touches dialogs owned by the Morrowind PID we were told about.
# A blind "close every #32770 on the desktop" would happily dismiss the user's own dialogs.
param(
    [int]$DurationSec = 900,
    [int]$PollMs = 1000
)

Add-Type @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class Win {
    public delegate bool EnumProc(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
    [DllImport("user32.dll")] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll")] public static extern IntPtr PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
}
"@

$WM_KEYDOWN = 0x0100
$WM_KEYUP   = 0x0101
$VK_RETURN  = 0x0D
$VK_SPACE   = 0x20

$deadline = (Get-Date).AddSeconds($DurationSec)
$seen = @{}

while ((Get-Date) -lt $deadline) {
    $procs = @(Get-Process Morrowind -ErrorAction SilentlyContinue)
    if ($procs.Count -eq 0) { Start-Sleep -Milliseconds $PollMs; continue }
    $pids = @{}
    foreach ($p in $procs) { $pids[[uint32]$p.Id] = $true }

    $cb = [Win+EnumProc]{
        param($h, $l)
        if (-not [Win]::IsWindowVisible($h)) { return $true }
        $wpid = [uint32]0
        [void][Win]::GetWindowThreadProcessId($h, [ref]$wpid)
        if (-not $pids.ContainsKey($wpid)) { return $true }

        $cls = New-Object System.Text.StringBuilder 256
        [void][Win]::GetClassName($h, $cls, 256)
        # #32770 is the Win32 dialog class. Morrowind's own render window is not one, so this
        # cannot accidentally post Enter into gameplay.
        if ($cls.ToString() -ne "#32770") { return $true }

        $txt = New-Object System.Text.StringBuilder 512
        [void][Win]::GetWindowTextW($h, $txt, 512)
        $key = "$h"
        if (-not $seen.ContainsKey($key)) {
            Write-Output ("[dismiss] dialog '{0}' (hwnd {1}) -> Enter+Space" -f $txt.ToString(), $h)
            $seen[$key] = $true
        }
        foreach ($vk in @($VK_RETURN, $VK_SPACE)) {
            [void][Win]::PostMessage($h, $WM_KEYDOWN, [IntPtr]$vk, [IntPtr]0)
            [void][Win]::PostMessage($h, $WM_KEYUP,   [IntPtr]$vk, [IntPtr]0)
        }
        return $true
    }
    [void][Win]::EnumWindows($cb, [IntPtr]::Zero)
    Start-Sleep -Milliseconds $PollMs
}
Write-Output "[dismiss] watcher done"
