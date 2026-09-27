# Marks a 32-bit PE image LARGE ADDRESS AWARE (IMAGE_FILE_LARGE_ADDRESS_AWARE, 0x0020 in the COFF
# header's Characteristics), so it can use 4 GB of address space on 64-bit Windows.
#
# Replaces 3rdparty\4gb_patch\4gb_patch.exe in the RELEASE packaging step: that external tool's
# manifest demands admin rights, so every package build raised a UAC prompt, for a one-bit edit.
# The PE checksum is left alone on purpose: Windows verifies it only for drivers and boot images.
#
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools\set-laa.ps1 <path to exe>
param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'

$bytes = [System.IO.File]::ReadAllBytes($Path)
if ($bytes.Length -lt 0x40 -or $bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) { throw "$Path is not a PE image (no MZ header)" }
$pe = [BitConverter]::ToInt32($bytes, 0x3C)
if ($pe -lt 0 -or $pe + 24 -gt $bytes.Length -or [BitConverter]::ToUInt32($bytes, $pe) -ne 0x00004550) {
    throw "$Path has no PE signature at 0x$('{0:X}' -f $pe)"
}
$chOff = $pe + 4 + 18                       # 'PE\0\0', then Machine..SizeOfOptionalHeader = 18 bytes
$ch = [BitConverter]::ToUInt16($bytes, $chOff)
if ($ch -band 0x0020) {
    Write-Host "set-laa: $Path is already large address aware"
    exit 0
}
$ch = $ch -bor 0x0020
$bytes[$chOff] = [byte]($ch -band 0xFF)
$bytes[$chOff + 1] = [byte](($ch -shr 8) -band 0xFF)
[System.IO.File]::WriteAllBytes($Path, $bytes)
Write-Host "set-laa: $Path marked large address aware"
