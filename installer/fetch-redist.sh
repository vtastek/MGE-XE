#!/bin/bash
# Fetches the runtimes installer/mgexe.nsi bundles into a folder OUTSIDE the repo (default
# ../buildtools/redist) and refuses any file that is not validly signed by Microsoft.
#   VC_redist.x86.exe / VC_redist.x64.exe   current VC++ 2015-2022 (aka.ms links)
#   dx/                                     DXSETUP + the d3dx9_43 / D3DCompiler_43 cabs, x86 AND x64,
#                                           from the DirectX June 2010 redist
#   vc_redist_2010.x86.exe                  for SlimDX; Microsoft no longer hosts a stable link, so
#                                           pass upstream's MGE XE 0.18.0 installer to extract it:
#
# usage: installer/fetch-redist.sh [redist dir] [path to MGEXE-0.18.0-installer.exe]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${1:-$REPO/../buildtools/redist}"
UPSTREAM="${2:-}"
SEVENZIP="/mnt/c/Program Files/7-Zip/7z.exe"
mkdir -p "$R/dx" "$R/dl"

curl -sSL -o "$R/VC_redist.x86.exe" https://aka.ms/vs/17/release/vc_redist.x86.exe
curl -sSL -o "$R/VC_redist.x64.exe" https://aka.ms/vs/17/release/vc_redist.x64.exe
[ -f "$R/dl/directx_Jun2010_redist.exe" ] || curl -sSL -o "$R/dl/directx_Jun2010_redist.exe" \
    "https://download.microsoft.com/download/8/4/A/84A35BF1-DAFE-4AE8-82AF-AD2AE20B6B14/directx_Jun2010_redist.exe"
(cd "$R/dx" && "$SEVENZIP" e -y "$(wslpath -w "$R/dl/directx_Jun2010_redist.exe")" \
    DSETUP.dll DXSETUP.exe dsetup32.dll dxdllreg_x86.cab dxupdate.cab \
    Jun2010_d3dx9_43_x86.cab Jun2010_d3dx9_43_x64.cab \
    Jun2010_D3DCompiler_43_x86.cab Jun2010_D3DCompiler_43_x64.cab >/dev/null)
if [ -n "$UPSTREAM" ]; then
    (cd "$R" && "$SEVENZIP" e -y "$(wslpath -w "$UPSTREAM")" '_redist_\vc_redist_2010.x86.exe' >/dev/null)
fi

# Every executable must carry a valid Microsoft signature.
powershell.exe -NoProfile -Command "
  \$bad = 0
  Get-ChildItem -Recurse '$(wslpath -w "$R")' -Include *.exe,*.dll | Where-Object { \$_.FullName -notlike '*\dl\*' } | ForEach-Object {
    \$s = Get-AuthenticodeSignature \$_.FullName
    \$ok = (\$s.Status -eq 'Valid') -and (\$s.SignerCertificate.Subject -like 'CN=Microsoft Corporation*')
    '{0,-6} {1}' -f (\$(if (\$ok) {'OK'} else {'BAD'})), \$_.Name
    if (-not \$ok) { \$bad++ }
  }
  exit \$bad"
