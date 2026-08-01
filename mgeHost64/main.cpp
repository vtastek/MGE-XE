#include "support/winheader.h"
#include "support/log.h"
#include "mge/configuration.h"
#include "ipc/server.h"
#include "forgerender.h"
#include "terrain.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

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



int main(int argc, char** argv) {

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
		return ForgeRender::sceneProbe() ? 0 : 1;
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
	const int parsed = std::sscanf(GetCommandLineA(), "%p %p %p %p %p %p %p",
		&sharedMem, &clientProcess, &rpcStartEvent, &rpcCompleteEvent,
		&geomSharedMem, &geomRpcStartEvent, &geomRpcCompleteEvent);
	if (parsed != 7 && parsed != 4) {
		LOG::logline("Expected handles not found on command line (parsed %d)", parsed);
		LOG::flush();
		return 1;
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
		geomSharedMem, geomRpcStartEvent, geomRpcCompleteEvent);
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