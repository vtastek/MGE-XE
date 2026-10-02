#include "support/winheader.h"
#include "support/log.h"
#include "mge/configuration.h"
#include "ipc/server.h"
#include "forgerender.h"
#include "terrain.h"
#include "knobs.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

// DirectX 12 Agility SDK loader exports. The OS d3d12.dll looks these up in the host
// EXE at startup; without them it ignores the D3D12Core.dll we ship beside the exe and
// falls back to the older system runtime — which mismatches the Agility headers The Forge
// compiles against. Version 715 = the Agility SDK build vendored in The Forge (matches
// D3D12_AGILITY_SDK_VERSION). Path "" = load D3D12Core.dll from the exe's own directory.
// (Normally emitted by The Forge's DEFINE_APPLICATION_MAIN macro, which we don't use.)
extern "C" {
	__declspec(dllexport) extern const UINT D3D12SDKVersion = 715;
	__declspec(dllexport) extern const char* D3D12SDKPath = u8"";
}



// ─── OPT-IN: LET AN API PROXY IN THE EXE'S OWN FOLDER LOAD (ReShade, OptiScaler, …) ─────────────
//
// ⚠⚠ WHY THIS FUNCTION HAS TO EXIST AT ALL. The Forge loads the two graphics DLLs with an EXPLICIT
// system-directory-only search:
//
//     gD3D12dll = LoadLibraryExA("d3d12.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
//     gDXGIdll  = LoadLibraryExA("dxgi.dll",  NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
//                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Direct3D12.c
//
// so a proxy DLL sitting beside mgeHost64.exe is bypassed by construction and can NEVER be reached.
// Measured 2026-09-02: an install with ReShade 6.8.0 installed as `d3d12.dll` right next to the host
// produced not one line of ReShade output across many host runs, because the host never opened that
// file.
//
// ⚠ WHY THE FIX IS HERE AND NOT IN The Forge. Patching Direct3D12.c would work and would be one
// word, but it is a VENDORED third-party file and every such edit is a merge conflict forever
// ([[project_forge_fork_merge_trap]]). Windows resolves an already-loaded module by its BASE NAME
// before it consults any search path, so loading `d3d12.dll` ourselves FIRST — with the default
// search order, which does include the application directory — means The Forge's later
// SYSTEM32-only call finds the module already present and returns that same handle. The proxy wins
// without one line of The Forge changing.
//
// ⚠⚠ OPT-IN, AND IT STAYS OPT-IN. This injects arbitrary third-party code into the render host, and
// the host is the process that owns the frame: a bad proxy takes the renderer down rather than
// degrading. It is also useless by default — nothing ships a proxy in that folder — so the cost of
// having it on would be pure risk for no gain.
//
// ⚠ IT LOGS WHICH FILE ACTUALLY WON, by full path. "Did my proxy load" is otherwise exactly as
// unanswerable as "is DLSS running" was, and for the same reason: success and failure look
// identical from outside.
//
// The two knobs it reads are declared just above the function, not in forgerender.cpp — see the
// note there for why that is new and why it matters.
//

// The report, held until mgeHost64.log exists. See the note at the printf below.
static char gProxyReport[4][512] = {};
static int  gProxyReportCount = 0;
static void proxyReport(const char* fmt, ...) {
	if (gProxyReportCount < 4) {
		va_list a;
		va_start(a, fmt);
		std::vsnprintf(gProxyReport[gProxyReportCount], sizeof(gProxyReport[0]), fmt, a);
		va_end(a);
		++gProxyReportCount;
	}
}
// Emitted by main() the moment LOG::open has run. ⚠ THIS FUNCTION IS THE FIX FOR A REAL BLIND SPOT,
// not tidiness: the client spawns the host with CREATE_NO_WINDOW (ipc/client.cpp), so the host has
// NO CONSOLE in a real game session and everything this file printf'd went nowhere. The first
// attempt to use the proxy hook in game was therefore unanswerable — the one line that says whether
// it loaded was written to a stream that does not exist. Same lesson as the upscale backend's own
// logging, one process-lifecycle stage earlier.
static void flushProxyReport() {
	for (int i = 0; i < gProxyReportCount; ++i) {
		LOG::logline("%s", gProxyReport[i]);
	}
	if (gProxyReportCount) {
		LOG::flush();
	}
}

// ─── MAIN'S OWN TWO KNOBS ────────────────────────────────────────────────────────────────────────
// ⚠ THESE TWO ARE THE REASON THE KNOB REGISTRY EXISTS, IN MINIATURE. They are real knobs with a real
// consumer, and that consumer is in THIS file — but until the Phase 1 registry
// (tasks/forge-host-decomposition.md) a knob could only be DECLARED inside forgerender.cpp, beside
// the four static tables, because those tables are static arrays of pointers to file-scope objects.
// So these two could not be registered at all, and the cost was a hardcoded exception list in
// applyKnobSpec (`kOwnedByMain[]`) whose only job was to stop the parser calling them
// "UNKNOWN … ignored" in the same log that, four lines above, shows them having done their work. A
// warning that contradicts the evidence beside it is worse than no warning.
//
// They are now ordinary registered bool knobs owned by the file that reads them, and that exception
// list is gone. `--knob-dump` lists them with everything else, so "what knobs exist" finally has one
// answer instead of one answer plus a footnote.
//
// ⚠ THE ENV IS STILL READ BY strstr HERE, AND THAT IS NOT REDUNDANT. preloadGraphicsProxies() runs
// at the top of main(), long before ForgeRender::applyEnvOverrides() parses anything, because its
// whole purpose is to load DLLs before the renderer — and therefore before device creation — exists.
// Registration is what makes a knob NAMEABLE from the rest of the system; the strstr is what makes
// this one readable that early. They agree because they are the same spec string and the same name.
//
// ⚠ WHAT applyKnobSpec WRITES HERE IS THEREFORE A RECORD, NOT A CONTROL. By the time it runs main
// has already acted, so the value it stores equals what strstr found and the two can never disagree
// — but setting either one later cannot undo a LoadLibrary. That is also why neither has a panel
// row, the same rule `upscaleBackend` follows: a widget that silently does nothing after startup is
// worse than none.
static bool g_proxyDlls  = false;
static bool g_preloadNgx = false;
MGE_KNOB(g_proxyDlls,  "proxyDlls");
MGE_KNOB(g_preloadNgx, "preloadNgx");
static void preloadGraphicsProxies() {
	const char* env = std::getenv("MGE_HOST_KNOBS");
	// Set the registered knobs from the same spec string the renderer will parse later, so the value
	// --knob-dump prints is the value this function acted on rather than a plausible-looking default.
	g_proxyDlls  = (env && std::strstr(env, "proxyDlls=1")  != nullptr);
	g_preloadNgx = (env && std::strstr(env, "preloadNgx=1") != nullptr);
	if (!g_proxyDlls) {
		// ⚠ SAID OUT LOUD. "The knob is off" and "the knob is on and the load failed" produce the
		// same silence otherwise, and that ambiguity cost a whole round trip the first time.
		proxyReport(">> [proxy] disabled (MGE_HOST_KNOBS has no proxyDlls=1) — the graphics DLLs "
		            "come from The Forge's System32-only load, as they always have");
		return;
	}
	// ─── AND THE NGX RUNTIMES, WHEN ASKED ───────────────────────────────────────────────────────
	// ⚠⚠ THIS IS AN **ORDERING** FIX, and the ordering is the whole defect. An injected NGX overlay
	// (RenoDX's DLSS5 addon, and tools of that shape generally) scans the process for already-loaded
	// `nvngx_*` modules AT DEVICE INIT and installs its hooks on what it finds — once. This host
	// initialises NGX LAZILY, in buildOpaquePath on the first renderScene, which is long after the
	// device exists. So the scan runs against a process that has not touched NGX yet.
	//
	// Measured 2026-09-02, straight from the addon's own log:
	//     NGX module scan (loaded copies):
	//     <nothing>
	// and on a later device, only `nvngx_dlssnr.dll` — the one the addon itself had pre-loaded.
	// `nvngx_dlss.dll`, the super-resolution runtime this host actually uses, was never in the list.
	// The addon was therefore hooking nothing, which is exactly the reported symptom: the overlay
	// appears, its settings move, and the picture does not change.
	//
	// Loading the runtimes here — before the device, before any hook scan — puts them in the list.
	// It does not disturb NGX's own loading: it LoadLibrary's them by the same base name later and
	// Windows returns the module already resident, the same mechanism the d3d12 proxy above relies
	// on. Missing files are skipped in silence by design; a machine without frame generation has no
	// `nvngx_dlssg.dll` and that is not a problem to report.
	if (g_preloadNgx) {
		static const wchar_t* const kNgx[] = {
			L"nvngx_dlss.dll",     // super resolution — the one this host uses
			L"nvngx_dlssnr.dll",   // ray reconstruction / neural rendering
			L"nvngx_dlssg.dll",    // frame generation; unused here, loaded only to be scannable
		};
		int loaded = 0;
		for (const wchar_t* n : kNgx) {
			if (HMODULE m = LoadLibraryW(n)) {
				wchar_t p2[MAX_PATH] = L"";
				GetModuleFileNameW(m, p2, MAX_PATH);
				proxyReport(">> [proxy] preloaded %ls -> %ls", n, p2);
				++loaded;
			}
		}
		proxyReport(">> [proxy] NGX preload: %d module(s) resident BEFORE device creation, so an "
		            "injected overlay's module scan can see them. 0 here means none were found "
		            "beside the exe.", loaded);
	}

	static const wchar_t* const kNames[] = { L"d3d12.dll", L"dxgi.dll" };
	for (const wchar_t* name : kNames) {
		// Default search order: the EXE's directory is consulted before System32, which is exactly
		// the behaviour The Forge opted out of.
		HMODULE h = LoadLibraryW(name);
		if (!h) {
			proxyReport("!! [proxy] LoadLibrary(%ls) FAILED (error %lu) — the host falls back to "
			            "The Forge's System32 load", name, (unsigned long)GetLastError());
			continue;
		}
		wchar_t path[MAX_PATH] = L"";
		GetModuleFileNameW(h, path, MAX_PATH);
		// ⚠ BUFFERED, NOT LOGGED HERE — mgeHost64.log is opened a few lines into main() and this
		// has to run before the renderer, so LOG::write would silently no-op on a closed handle.
		// printf alone was the first attempt and it was WRONG for the case that matters: the client
		// spawns this process with CREATE_NO_WINDOW, so a game session has no stdout to read.
		// flushProxyReport() puts these in the log the moment it exists.
		proxyReport(">> [proxy] %ls -> %ls", name, path);
		std::printf("[proxy] %ls -> %ls\n", name, path);
	}
	std::fflush(stdout);
}

int main(int argc, char** argv) {
	// BEFORE every entry point below, because all of them reach the renderer and the renderer is
	// what loads the graphics DLLs. See preloadGraphicsProxies.
	preloadGraphicsProxies();

	// Standalone Forge bring-up probe (Milestone D1): run `mgeHost64.exe --forge-probe`
	// to validate the vendored Forge (D3D12) build initialises a device, with no
	// IPC handles / Morrowind needed.
	if (argc >= 2 && std::strcmp(argv[1], "--forge-probe") == 0) {
		return ForgeRender::probe() ? 0 : 1;
	}

	// Standalone Forge render probe (Milestone D3): render the hardcoded triangle
	// to a Forge-owned render target and verify the readback. No MW seam.
	if (argc >= 2 && std::strcmp(argv[1], "--forge-render") == 0) {
		return ForgeRender::renderTriangle() ? 0 : 1;
	}

	// Standalone shared-RT render probe (Milestone D4 host half): render the
	// triangle into a cross-process SHARED D3D12 render target + export an NT
	// handle. Proves the host side of the route-1 present seam.
	if (argc >= 2 && std::strcmp(argv[1], "--forge-render-shared") == 0) {
		return ForgeRender::renderTriangleShared() ? 0 : 1;
	}

	// Standalone Phase 1a distant-land probe: load + render host-owned DL from a synthetic
	// camera (no MW/IPC). MUST run with cwd = morrowind64 so the DL files resolve.
	if (argc >= 2 && std::strcmp(argv[1], "--forge-dl") == 0) {
		return ForgeRender::renderDistantLandProbe() ? 0 : 1;
	}

	// Standalone Phase 1b distant-statics probe: load + GPU-driven-render host-owned distant statics
	// (instancing + bindless + execute-indirect) over the land. MUST run with cwd = morrowind64.
	if (argc >= 2 && std::strcmp(argv[1], "--forge-statics") == 0) {
		return ForgeRender::renderDistantStaticsProbe() ? 0 : 1;
	}

	// Standalone interactive world viewer: a real window showing host-owned distant land + statics
	// you fly around (WASD + arrows, ESC quit). No Morrowind/IPC — isolates Forge from integration.
	// MUST run with cwd = morrowind64.
	if (argc >= 2 && std::strcmp(argv[1], "--forge-view") == 0) {
		return ForgeRender::worldViewer() ? 0 : 1;
	}

	// Standalone M1c opaque scene-path probe: init + uploadGeometry + renderScene with a
	// dummy mesh, so a buildOpaquePath/draw crash is visible on stdout (no MW/IPC).
	if (argc >= 2 && std::strcmp(argv[1], "--forge-scene") == 0) {
		// `--forge-scene pix` arms a PIX programmatic GPU capture around one renderScene
		// (the headless probe never Presents, so PIX's hotkey capture can't trigger).
		if (argc >= 3 && std::strcmp(argv[2], "pix") == 0) {
			ForgeRender::enablePixCapture();   // loads WinPixGpuCapturer.dll BEFORE device init
		}
		// `--forge-scene rdoc` arms a RenderDoc capture instead (better descriptor-table inspection).
		if (argc >= 3 && std::strcmp(argv[2], "rdoc") == 0) {
			ForgeRender::enableRdocCapture(true);  // loads renderdoc.dll BEFORE device init
		}
		// `--forge-scene 1x` / `2x` / `4x` / `8x` picks the MSAA arm. Since M0
		// (tasks/forge-upscale.md) the sample count selects between two genuinely different
		// pipelines — 1x now has its own scene-referred staging target and its own resolve_sc1
		// variant — and the probe's flat fullscreen triangle has no edge for AA to act on, so the
		// arms must agree pixel for pixel. That equality is the M0 gate and it needs no Morrowind.
		unsigned probeSamples = 4;
		if (argc >= 3) {
			if (std::strcmp(argv[2], "1x") == 0)      { probeSamples = 1; }
			else if (std::strcmp(argv[2], "2x") == 0) { probeSamples = 2; }
			else if (std::strcmp(argv[2], "4x") == 0) { probeSamples = 4; }
			else if (std::strcmp(argv[2], "8x") == 0) { probeSamples = 8; }
		}
		// ⚠ THE PROBE NEEDS THE ENV KNOBS TOO, and it returns long before the call below that
		// normally applies them. Without this, `MGE_HOST_KNOBS=upscaleEnable=1 --forge-scene 1x`
		// silently runs the DEFAULT arm and reports it as the overridden one — a run labelled as an
		// A/B whose two arms are identical, which is precisely the failure the knob table's own
		// header says it exists to prevent. Found running M1 4b's verification steps 3 and 5, both
		// of which are probe runs with an env knob set.
		//
		// A SECOND CALL rather than moving the one below, deliberately: that one runs AFTER
		// LOG::open, and its `>> [forge] MGE_HOST_KNOBS = ...` line is what makes a normal run's log
		// carry the arm it was measured in. Hoisting it would drop that line from every real session
		// to serve the probe. The probe has no MGE log at all (it returns before LOG::open, which is
		// also why `[scenefmt]` is printf'd), so its evidence is stdout — which is why the upscale
		// backend's ready / DECLINED lines go to both. Applying twice is harmless: the parse writes
		// the same values into the same knobs.
		ForgeRender::applyEnvOverrides();
		return ForgeRender::sceneProbe(probeSamples) ? 0 : 1;
	}

	// Standalone knob dump: print every MGE_HOST_KNOBS knob (kind, name, value, clamp) sorted by
	// name, with no GPU, no IPC and no Morrowind — so the knob table can be diffed across a build
	// from a shell redirect. Same standalone shape as --terrain-census, and it exists for the same
	// reason: a property that needs a game session to check is a property nobody checks.
	//
	// ⚠ RUNS BEFORE applyEnvOverrides ON PURPOSE. This prints the BUILD DEFAULTS, which is what a
	// refactor's before/after comparison has to be about; a dump taken after the environment had a
	// say would differ between two identical builds launched from two different shells. Set
	// MGE_HOST_KNOBS and you will still see the defaults here — that is the contract, not a bug.
	if (argc >= 2 && std::strcmp(argv[1], "--knob-dump") == 0) {
		return ForgeRender::dumpKnobs() > 0 ? 0 : 1;
	}

	// Standalone terrain census (tasks/forge-terrain.md T0): parse every plugin's LAND/LTEX
	// records and print the census, with no GPU, no IPC and no Morrowind — so the loader can be
	// verified (cell count, extent, texture count, parse time) without driving the game.
	// MUST run with cwd = the install dir.
	if (argc >= 2 && std::strcmp(argv[1], "--terrain-census") == 0) {
		LOG::open("mgeHost64_terrain.log");
		Terrain::beginLoadAsync();
		const bool ok = Terrain::waitLoaded(600000);
		LOG::flush();
		return (ok && Terrain::cellCount() > 0) ? 0 : 1;
	}

	LOG::open("mgeHost64.log");
	LOG::logline("Host process started");
	// What the pre-main proxy preload did, now that there is somewhere to say it.
	flushProxyReport();

	// Dev-knob overrides from the environment, BEFORE init so a knob read during bring-up sees the
	// override rather than the default. Logged, so a run's log carries the arm it was measured in.
	ForgeRender::applyEnvOverrides();

	// GPU capture, BEFORE any device creation (renderdoc.dll has to hook d3d12 first). Attach-only
	// by default: if the RenderDoc UI launched or injected us the dll is already in the process and
	// numpad 0 works, and if it is not this costs one failed GetModuleHandle — a shipped install
	// never pulls the hooks into a play session. MGE_RDOC=1 opts into the LoadLibrary for the case
	// where the host was spawned by the client rather than launched under the UI.
	{
		const bool allowLoad = std::getenv("MGE_RDOC") != nullptr;
		if (ForgeRender::enableRdocCapture(allowLoad)) {
			LOG::logline(">> RenderDoc attached — numpad 0 in-game captures the next host frame");
		}
	}

	// Host-owned terrain (tasks/forge-terrain.md): read every plugin's LAND records NOW, on a
	// background thread, so the parse overlaps MW's own load and the client handshake below.
	// cwd is the install dir, so Morrowind.ini and Data Files resolve relative. No disk artifact,
	// no MGEgui bake, nothing to regenerate when the mod list changes.
	Terrain::beginLoadAsync();

	HANDLE sharedMem = INVALID_HANDLE_VALUE;
	HANDLE clientProcess = INVALID_HANDLE_VALUE;
	HANDLE rpcStartEvent = INVALID_HANDLE_VALUE;
	HANDLE rpcCompleteEvent = INVALID_HANDLE_VALUE;
	// Dedicated geometry channel handles (second shared-mem + start/complete events). The
	// client always passes all 7 now; the 4-handle form is kept for forward/back-compat.
	HANDLE geomSharedMem = nullptr;
	HANDLE geomRpcStartEvent = nullptr;
	HANDLE geomRpcCompleteEvent = nullptr;
	// Async texture stream channel (third shared-mem + start/complete events); 10-handle form only.
	HANDLE streamSharedMem = nullptr;
	HANDLE streamRpcStartEvent = nullptr;
	HANDLE streamRpcCompleteEvent = nullptr;
	const int parsed = std::sscanf(GetCommandLineA(), "%p %p %p %p %p %p %p %p %p %p",
		&sharedMem, &clientProcess, &rpcStartEvent, &rpcCompleteEvent,
		&geomSharedMem, &geomRpcStartEvent, &geomRpcCompleteEvent,
		&streamSharedMem, &streamRpcStartEvent, &streamRpcCompleteEvent);
	if (parsed != 10 && parsed != 7 && parsed != 4) {
		LOG::logline("Expected handles not found on command line (parsed %d)", parsed);
		LOG::flush();
		return 1;
	}
	if (parsed < 10) {
		streamSharedMem = streamRpcStartEvent = streamRpcCompleteEvent = nullptr;
	}
	if (parsed == 4) {
		geomSharedMem = geomRpcStartEvent = geomRpcCompleteEvent = nullptr;
	}

#ifdef _DEBUG
	while (!IsDebuggerPresent()) {
		Sleep(100);
	}
#endif

	Configuration.LoadSettings();
	if (!IPC::initImports()) {
		LOG::logline("!! Required memory mapping APIs are not available");
		LOG::flush();
		return 4;
	}

	IPC::Server server(sharedMem, clientProcess, rpcStartEvent, rpcCompleteEvent,
		geomSharedMem, geomRpcStartEvent, geomRpcCompleteEvent,
		streamSharedMem, streamRpcStartEvent, streamRpcCompleteEvent);
	if (!server.init()) {
		LOG::logline("!! Server initialization failed");
		LOG::flush();
		return 2;
	}
	if (!server.listen()) {
		LOG::logline("!! Server listen failed");
		LOG::flush();
		return 3;
	}

	LOG::flush();
	return 0;
}