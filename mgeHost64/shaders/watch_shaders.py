#!/usr/bin/env python3
"""
mgeHost64 FSL hot-reload watcher  (the compile + deploy half of hot-reload).

The HOST already rebuilds its compute pipelines when gtao.comp_0.dxil changes on disk. But editing
a .fsl SOURCE does nothing until it is COMPILED (fsl.py) and DEPLOYED to the game's DIRECT3D12 dir.
This script is exactly those two missing steps, on every save:

    edit .fsl  ->  [this: fsl.py compile -> atomic deploy]  ->  host auto-reload  ->  live

RUN WITH WINDOWS PYTHON (fsl.py shells out to the bundled DXC; WSL python will NOT work):

    & "C:\\Users\\devbox\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" `
      "C:\\projects\\mgexe\\MGE-XE\\mgeHost64\\shaders\\watch_shaders.py"

Leave it running in its own terminal while the game is up. It prints what it does on EVERY save, so
if a save produces no "[compile] ok / [deploy] ok" lines, detection failed; if it prints a compile
error, the shader is broken. Ctrl+C to stop.
"""
import os
import sys
import time
import shutil
import traceback
import subprocess

ROOT      = r"C:\projects\mgexe\MGE-XE"
SHADERS   = os.path.join(ROOT, "mgeHost64", "shaders")
FSL_DIR   = os.path.join(SHADERS, "FSL")
GEN_DIR   = os.path.join(SHADERS, "gen")
BIN_DIR   = os.path.join(SHADERS, "bin")
BIN_D3D12 = os.path.join(BIN_DIR, "DIRECT3D12")
FSL_PY    = os.path.join(ROOT, "3rdparty", "The-Forge", "Common_3", "Tools",
                         "ForgeShadingLanguage", "fsl.py")
LIST_REL  = os.path.join("FSL", "shaders.list")     # relative to SHADERS (fsl.py cwd)
DEPLOY    = r"C:\mgem\morrowind64\DIRECT3D12"

# The host watches this file's mtime; copy it LAST + atomically so its deps are already in place.
TRIGGER      = "gtao.comp_0.dxil"
POLL_SECONDS = 0.4


def log(msg):
    sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def preflight():
    ok = True
    log("watcher starting")
    log("  python : %s" % sys.executable)
    for label, p, isdir in (("fsl.py", FSL_PY, False), ("FSL dir", FSL_DIR, True),
                            ("bin/D3D12", BIN_D3D12, True), ("deploy", DEPLOY, True)):
        exists = os.path.isdir(p) if isdir else os.path.isfile(p)
        log("  %-9s: %s  %s" % (label, p, "OK" if exists else "*** MISSING ***"))
        if not exists and not (label == "deploy"):   # deploy dir we can create
            ok = False
    if not os.path.isdir(DEPLOY):
        os.makedirs(DEPLOY, exist_ok=True)
        log("  created deploy dir")
    files = watched_files()
    log("  watching %d source files in FSL/ (.fsl/.h/.list)" % len(files))
    if not ok:
        log("PREFLIGHT FAILED — fix the *** MISSING *** path(s) above. Watcher will idle.")
    return ok


def watched_files():
    out = []
    try:
        for name in os.listdir(FSL_DIR):
            if name.endswith((".fsl", ".h", ".list")):
                out.append(os.path.join(FSL_DIR, name))
    except OSError as e:
        log("listdir failed: %s" % e)
    return out


def snapshot():
    s = {}
    for f in watched_files():
        try:
            s[f] = os.path.getmtime(f)
        except OSError:
            pass
    return s


def compile_shaders():
    cmd = [sys.executable, FSL_PY, "-d", GEN_DIR, "-b", BIN_DIR,
           "-l", "DIRECT3D12", "--compile", LIST_REL]
    r = subprocess.run(cmd, cwd=SHADERS, capture_output=True, text=True)
    if r.returncode != 0:
        log("[compile] FAILED (returncode %d) ----------------------------------" % r.returncode)
        tail = (r.stdout or "") + (r.stderr or "")
        sys.stdout.write(tail[-3000:] + "\n")
        sys.stdout.write("--------------------------------------------------------------\n")
        sys.stdout.flush()
        return False
    log("[compile] ok")
    return True


def atomic_copy(src, dst):
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)   # atomic on the same volume


def deploy():
    files = [f for f in os.listdir(BIN_D3D12)
             if os.path.isfile(os.path.join(BIN_D3D12, f))]
    n = 0
    for f in sorted(files):           # everything except the trigger first
        if f == TRIGGER:
            continue
        atomic_copy(os.path.join(BIN_D3D12, f), os.path.join(DEPLOY, f))
        n += 1
    trig = os.path.join(BIN_D3D12, TRIGGER)
    if os.path.isfile(trig):          # then the trigger last (host watches its mtime)
        atomic_copy(trig, os.path.join(DEPLOY, TRIGGER))
        sz = os.path.getsize(os.path.join(DEPLOY, TRIGGER))
        log("[deploy] ok — %d files + %s (%d bytes). Host reloads within a frame." % (n, TRIGGER, sz))
    else:
        log("[deploy] WARNING: %s not found in bin — did the compile actually run?" % TRIGGER)


def main():
    preflight()
    last = snapshot()
    log("ready — edit + save a .fsl in FSL/ to trigger a rebuild.")
    while True:
        time.sleep(POLL_SECONDS)
        try:
            cur = snapshot()
            if cur == last:
                continue
            changed = [os.path.basename(f) for f in cur if last.get(f) != cur[f]]
            last = cur
            log("change detected: %s" % ", ".join(changed))
            if compile_shaders():
                deploy()
        except Exception:
            log("watcher loop error:")
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\n[watch] stopped.\n")
