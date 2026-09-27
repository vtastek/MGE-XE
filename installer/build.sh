#!/bin/bash
# Builds the two release archives from a CLEAN bin/MSVC-Packaged-Release:
#   bin/release/MGE XE Manual Install-<ver>.7z   the package as-is (no runtimes; extract + run MWSE-Update)
#   bin/release/MGE XE Installer-<ver>.7z        MGEXE-<ver>-installer.exe, which bundles the VC++/DirectX
#                                                runtimes and runs MWSE-Update (installer/mgexe.nsi)
# Same shapes as upstream's 0.18.0 Nexus files.
#
# usage: installer/build.sh <version> [redist dir] [makensis.exe]
#   redist dir defaults to ../buildtools/redist (fill it with installer/fetch-redist.sh)
#   makensis   defaults to ../buildtools/nsis-3.10/makensis.exe (the portable NSIS zip)
set -euo pipefail

VER="${1:?usage: build.sh <version> [redistdir] [makensis]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REDIST="${2:-$REPO/../buildtools/redist}"
MAKENSIS="${3:-$REPO/../buildtools/nsis-3.10/makensis.exe}"
SEVENZIP="/mnt/c/Program Files/7-Zip/7z.exe"
PKG="$REPO/bin/MSVC-Packaged-Release"
OUT="$REPO/bin/release"
WORK="$REPO/bin/installer-work"

[ -f "$PKG/mgeHost64.exe" ] || { echo "no package at $PKG — build RELEASE + the Forge runtime step first"; exit 1; }
for f in VC_redist.x86.exe VC_redist.x64.exe vc_redist_2010.x86.exe dx/DXSETUP.exe; do
    [ -f "$REDIST/$f" ] || { echo "missing $REDIST/$f — run installer/fetch-redist.sh"; exit 1; }
done
# Dev-only files must never ship (same list release.yml enforces).
for f in mgeXE_verbose_log.txt mgeHostPanel.ini mgeXE_fslwatch.txt license.txt; do
    [ -e "$PKG/$f" ] && { echo "DEV/FOREIGN FILE IN PACKAGE: $f"; exit 1; }
done

rm -rf "$WORK"
mkdir -p "$WORK" "$OUT"

# Uninstall list: every packaged file, then every packaged folder deepest-first with a plain RMDir
# (removes it only if empty — Data Files and its subfolders are shared with other mods).
{
    (cd "$PKG" && find . -type f | sed 's|^\./||' | sort) | while IFS= read -r f; do
        printf 'Delete "$INSTDIR\\%s"\n' "${f//\//\\}"
    done
    (cd "$PKG" && find . -mindepth 1 -type d | sed 's|^\./||' | awk '{ print length($0) "\t" $0 }' | sort -rn | cut -f2-) \
    | while IFS= read -r d; do
        printf 'RMDir "$INSTDIR\\%s"\n' "${d//\//\\}"
    done
} > "$WORK/uninst_files.nsh"

EXE="MGEXE-$VER-installer.exe"
"$MAKENSIS" -V2 -INPUTCHARSET UTF8 \
    "-DVERSION=$VER" \
    "-DPKGDIR=$(wslpath -w "$PKG")" \
    "-DREDISTDIR=$(wslpath -w "$REDIST")" \
    "-DLICENSEFILE=$(wslpath -w "$REPO/license.txt")" \
    "-DUNINST_LIST=$(wslpath -w "$WORK/uninst_files.nsh")" \
    "-DOUTFILE=$(wslpath -w "$WORK/$EXE")" \
    "$(wslpath -w "$REPO/installer/mgexe.nsi")"

MANUAL="$OUT/MGE XE Manual Install-$VER.7z"
INST="$OUT/MGE XE Installer-$VER.7z"
rm -f "$MANUAL" "$INST"
(cd "$PKG" && "$SEVENZIP" a -t7z -mx=9 -ms=on "$(wslpath -w "$MANUAL")" '*' >/dev/null)
(cd "$WORK" && "$SEVENZIP" a -t7z -mx=9 "$(wslpath -w "$INST")" "$EXE" >/dev/null)
"$SEVENZIP" t "$(wslpath -w "$MANUAL")" >/dev/null
"$SEVENZIP" t "$(wslpath -w "$INST")" >/dev/null
ls -la "$OUT"
