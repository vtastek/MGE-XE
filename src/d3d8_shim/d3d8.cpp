#define STRICT
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objbase.h>
#include <cstdio>

// Tiny forwarding shim. Morrowind loads game-dir d3d8.dll; only inside
// Morrowind.exe does this load the real MGE DLL (mgecore.dll). The launcher
// and Construction Set never execute any MGE/SharedSE code, so their loader
// can't be killed by CRT static initializers touching Morrowind.exe addresses.

typedef void* (__stdcall* D3DProc)(UINT);
typedef HRESULT (__stdcall* DInputProc)(HINSTANCE, DWORD, REFIID, void**, void*);

static HMODULE mgeCore;
static D3DProc mgeDirect3DCreate8;
static DInputProc mgeDirectInput8Create;

static FARPROC getSystemProc(const char* lib, const char* funcname) {
    // Resolve from the system32 copy by full path; a bare name would recurse
    // onto the game-dir shims (this DLL / the dinput8 shim).
    char syspath[MAX_PATH], path[MAX_PATH];
    GetSystemDirectoryA(syspath, sizeof(syspath));

    int str_sz = std::snprintf(path, sizeof(path), "%s\\%s", syspath, lib);
    if (str_sz >= sizeof(path)) {
        return NULL;
    }

    HMODULE dll = LoadLibraryA(path);
    if (dll == NULL) {
        return NULL;
    }

    return GetProcAddress(dll, funcname);
}

extern "C" BOOL __stdcall DllMain(HANDLE hModule, DWORD reason, void* unused) {
    if (reason != DLL_PROCESS_ATTACH) {
        return TRUE;
    }

    if (GetModuleHandleA("Morrowind.exe")) {
        // Load at import time to preserve MGE init ordering; MWInitPatch and
        // MWSE injection must run before any Morrowind code executes.
        mgeCore = LoadLibraryA("mgecore.dll");
        if (mgeCore) {
            mgeDirect3DCreate8 = (D3DProc)GetProcAddress(mgeCore, "Direct3DCreate8");
            mgeDirectInput8Create = (DInputProc)GetProcAddress(mgeCore, "DirectInput8Create");
        }
    } else if (GetModuleHandleA("TES Construction Set.exe")) {
        // Load extender for CS, it injects by itself
        LoadLibraryA("CSSE.dll");
    }

    return TRUE;
}

extern "C" void* __stdcall FakeDirect3DCreate(UINT version) {
    if (mgeDirect3DCreate8) {
        return mgeDirect3DCreate8(version);
    }

    D3DProc func = (D3DProc)getSystemProc("d3d8.dll", "Direct3DCreate8");
    return func ? func(version) : NULL;
}

extern "C" HRESULT __stdcall FakeDirectInputCreate(HINSTANCE a, DWORD b, REFIID c, void** d, void* e) {
    if (mgeDirectInput8Create) {
        return mgeDirectInput8Create(a, b, c, d, e);
    }

    DInputProc func = (DInputProc)getSystemProc("dinput8.dll", "DirectInput8Create");
    return func ? func(a, b, c, d, e) : E_FAIL;
}
