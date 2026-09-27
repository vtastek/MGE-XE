; MGE XE installer (NSIS 3, Unicode). Built by installer/build.sh — do not run makensis by hand
; unless you pass the same /D defines:
;   VERSION    e.g. 0.20.4
;   PKGDIR     the clean release package (bin\MSVC-Packaged-Release) — installed as-is
;   REDISTDIR  the redistributables fetched by installer/fetch-redist.sh
;   OUTFILE    the installer exe to write
;   LICENSEFILE, UNINST_LIST   the repo license.txt and build.sh's generated delete list
;
; What it adds over the manual-install archive (which is PKGDIR zipped): the Morrowind folder is
; found for you, the runtimes the binaries import are installed, MWSE is fetched with the bundled
; MWSE-Update, and an uninstaller is registered. The runtimes, from the shipped binaries' imports:
;   VC++ 2015-2022 x86   d3d8.dll, mgecore.dll, dinput8.dll, mge3\MGEfuncs.dll
;   VC++ 2015-2022 x64   mgeHost64.exe (VCRUNTIME140_1 — x64 only)
;   DirectX Jun 2010     d3dx9_43 x86 (mgecore, MGEfuncs) AND x64 (mgeHost64); D3DCompiler_43
;   VC++ 2010 x86        SlimDX.dll (MGEXEgui)

Unicode true
SetCompressor /SOLID lzma
RequestExecutionLevel admin

!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"

!ifndef VERSION
  !error "pass /DVERSION=x.y.z"
!endif

Name "MGE XE ${VERSION}"
OutFile "${OUTFILE}"
BrandingText "MGE XE ${VERSION}"
InstallDir "$PROGRAMFILES32\Steam\steamapps\common\Morrowind"

!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\MGE XE"

!define MUI_ABORTWARNING
!define MUI_COMPONENTSPAGE_SMALLDESC
!define MUI_WELCOMEPAGE_TEXT "This installs MGE XE ${VERSION} into your Morrowind folder.$\r$\n$\r$\nMorrowind Code Patch is required for MWSE and its mods; install it first if you have not.$\r$\n$\r$\nClose Morrowind and the Construction Set before continuing."
!insertmacro MUI_PAGE_WELCOME
; The repo's license, NOT a file in PKGDIR: Morrowind's own folder already has a license.txt
; (Bethesda's), so MGE XE never installs one there.
!insertmacro MUI_PAGE_LICENSE "${LICENSEFILE}"
!insertmacro MUI_PAGE_COMPONENTS
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE CheckMorrowindDir
!define MUI_DIRECTORYPAGE_TEXT_TOP "Select your Morrowind folder: the one that contains Morrowind.exe."
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TEXT "MGE XE is installed.$\r$\n$\r$\nRun MGEXEgui.exe in your Morrowind folder to choose settings and generate distant land before playing."
!define MUI_FINISHPAGE_RUN "$INSTDIR\MGEXEgui.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open MGEXEgui now"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ── Find Morrowind ─────────────────────────────────────────────────────────────────────────────
; The launcher writes this key (NSIS is 32-bit, so HKLM reads the WOW6432Node view MW uses).
Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "MGE XE's renderer (mgeHost64.exe) needs 64-bit Windows."
    Abort
  ${EndIf}
  ReadRegStr $0 HKLM "Software\Bethesda Softworks\Morrowind" "Installed Path"
  ${If} $0 != ""
  ${AndIf} ${FileExists} "$0\Morrowind.exe"
    StrCpy $INSTDIR $0
  ${EndIf}
FunctionEnd

Function CheckMorrowindDir
  ${IfNot} ${FileExists} "$INSTDIR\Morrowind.exe"
    MessageBox MB_ICONEXCLAMATION "Morrowind.exe was not found in$\r$\n$INSTDIR$\r$\n$\r$\nSelect the folder Morrowind is installed in."
    Abort
  ${EndIf}
FunctionEnd

; ── Sections ───────────────────────────────────────────────────────────────────────────────────
Section "MGE XE" SecCore
  SectionIn RO
  SetOutPath "$INSTDIR"
  SetOverwrite on
  File /r "${PKGDIR}\*.*"

  WriteUninstaller "$INSTDIR\uninstall_MGEXE.exe"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "MGE XE ${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "MGE XE"
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\uninstall_MGEXE.exe"'
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
SectionEnd

Section "Visual C++ and DirectX runtimes" SecRuntimes
  SetOutPath "$PLUGINSDIR\redist"
  File "${REDISTDIR}\VC_redist.x86.exe"
  File "${REDISTDIR}\VC_redist.x64.exe"
  File "${REDISTDIR}\vc_redist_2010.x86.exe"
  File /r "${REDISTDIR}\dx"

  DetailPrint "Installing Visual C++ 2015-2022 runtime (x86)..."
  ExecWait '"$PLUGINSDIR\redist\VC_redist.x86.exe" /install /quiet /norestart'
  DetailPrint "Installing Visual C++ 2015-2022 runtime (x64)..."
  ExecWait '"$PLUGINSDIR\redist\VC_redist.x64.exe" /install /quiet /norestart'
  DetailPrint "Installing Visual C++ 2010 runtime (x86)..."
  ExecWait '"$PLUGINSDIR\redist\vc_redist_2010.x86.exe" /q /norestart'
  DetailPrint "Installing DirectX 9 components (June 2010)..."
  ExecWait '"$PLUGINSDIR\redist\dx\DXSETUP.exe" /silent'
  SetOutPath "$INSTDIR"
SectionEnd

Section "Download and update MWSE (internet)" SecMWSE
  DetailPrint "Running MWSE-Update (a console window opens; press a key there when it finishes)..."
  ExecWait '"$INSTDIR\MWSE-Update.exe"'
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore}     "MGE XE and its renderer. Required."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecRuntimes} "The Visual C++ runtimes (x86 and x64) and the DirectX 9 June 2010 components MGE XE needs. Safe to leave on: already-installed runtimes are skipped."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMWSE}     "Fetches the latest MWSE 2.1 so Lua mods and MGE XE's in-game options work. Needs Morrowind Code Patch."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ── Uninstall: exactly the files this installer put down (uninst_files.nsh, generated from
; PKGDIR by build.sh). Folders are removed only when empty — Data Files is shared with other mods.
Section "Uninstall"
  !include "${UNINST_LIST}"
  Delete "$INSTDIR\uninstall_MGEXE.exe"
  DeleteRegKey HKLM "${UNINST_KEY}"
SectionEnd
