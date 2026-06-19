#include "support/winheader.h"
#include "support/log.h"
#include "mge/configuration.h"
#include "ipc/server.h"
#include "forgerender.h"

#include <cstdio>
#include <cstring>

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

	LOG::open("mgeHost64.log");
	LOG::logline("Host process started");

	HANDLE sharedMem = INVALID_HANDLE_VALUE;
	HANDLE clientProcess = INVALID_HANDLE_VALUE;
	HANDLE rpcStartEvent = INVALID_HANDLE_VALUE;
	HANDLE rpcCompleteEvent = INVALID_HANDLE_VALUE;
	if (std::sscanf(GetCommandLineA(), "%p %p %p %p", &sharedMem, &clientProcess, &rpcStartEvent, &rpcCompleteEvent) != 4) {
		LOG::logline("Expected handles not found on command line");
		LOG::flush();
		return 1;
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

	IPC::Server server(sharedMem, clientProcess, rpcStartEvent, rpcCompleteEvent);
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