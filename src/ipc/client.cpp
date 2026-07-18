#include "client.h"
#include "support/log.h"

#include <cassert>
#include <cstdio>
#include <cstring>

// ideally this would go in beginRpc, but we can't put it there because we
// need to check that the previous command has finished before we start
// manipulating parameters
//
// ASYNC-FRAME WINDOW GUARD: between renderSceneKickoff and renderSceneFinish the
// RenderFrame RPC is deliberately left pending while the client does its own frame
// work. Any other RPC in that window would drain (steal) the RenderFrame completion
// and clobber the shared Parameters union mid-frame — a silent desync that surfaces
// as a 60s IPC timeout frames later. Fail LOUD instead: refuse the RPC and log.
#define WAIT_FOR_PREVIOUS_COMMAND { \
	if (m_frameWindowOpen) { \
		++m_windowRefusals; \
		LOG::logline("!! IPC RPC attempted inside the async RenderFrame window (would clobber the pending frame)"); \
		return false; \
	} \
	if (tryWaitForCompletion() != WakeReason::Complete) { \
		return false; \
	} \
}

namespace IPC {
	Client::Client() :
		m_process(INVALID_HANDLE_VALUE),
		m_sharedMem(INVALID_HANDLE_VALUE),
		m_rpcStartEvent(INVALID_HANDLE_VALUE),
		m_rpcCompleteEvent(INVALID_HANDLE_VALUE),
		m_ipcParameters(nullptr),
		m_isRpcPending(false),
		m_frameWindowOpen(false),
		m_windowRefusals(0),
		m_watcherProcess(INVALID_HANDLE_VALUE),
		m_watcherJob(NULL),
		m_geomSharedMem(INVALID_HANDLE_VALUE),
		m_geomRpcStartEvent(INVALID_HANDLE_VALUE),
		m_geomRpcCompleteEvent(INVALID_HANDLE_VALUE),
		m_geomParameters(nullptr),
		m_geomRpcPending(false)
	{
		m_geomWaitHandles[0] = INVALID_HANDLE_VALUE;
		m_geomWaitHandles[1] = INVALID_HANDLE_VALUE;
	}

	Client::~Client() {
		stopWatcher();
		if (m_process != INVALID_HANDLE_VALUE) {
			TerminateProcess(m_process, 0);
			CloseHandle(m_process);
			m_process = INVALID_HANDLE_VALUE;
		}

		if (m_ipcParameters != nullptr) {
			UnmapViewOfFile(m_ipcParameters);
			m_ipcParameters = nullptr;
		}

		CleanupHandle(m_sharedMem);
		CleanupHandle(m_rpcStartEvent);
		CleanupHandle(m_rpcCompleteEvent);

		if (m_geomParameters != nullptr) {
			UnmapViewOfFile(m_geomParameters);
			m_geomParameters = nullptr;
		}
		CleanupHandle(m_geomSharedMem);
		CleanupHandle(m_geomRpcStartEvent);
		CleanupHandle(m_geomRpcCompleteEvent);
	}

	bool Client::isServerActive() {
		if (m_process != INVALID_HANDLE_VALUE) {
			if (WaitForSingleObject(m_process, 0) == WAIT_OBJECT_0) {
				// the host process was started but is no longer running
				CloseHandle(m_process);
				m_process = INVALID_HANDLE_VALUE;
				return false;
			}

			return true;
		}

		return false;
	}

	void Client::startWatcher() {
		// Dev-only FSL hot-reload watcher: recompiles + deploys shaders on save so the host's
		// dxil-mtime auto-reload makes edit->live automatic. Two gates keep it out of every non-dev run:
		//   (1) the dev python + script paths must exist (never true in a shipped tree), AND
		//   (2) the running game must BE the install the watcher deploys into (C:\mgem\morrowind64 — see
		//       watch_shaders.py DEPLOY). So the RELEASE install (e.g. morrowinddonottouch) run on this
		//       same dev machine gets NO watcher and NO window.
		// In the dev install it launches with a MINIMIZED, non-activating console (SW_SHOWMINNOACTIVE):
		// its compile/deploy log is there when wanted but it never steals focus from the game or covers it.
		static const char* kPython = "C:\\Users\\devbox\\AppData\\Local\\Programs\\Python\\Python311\\python.exe";
		static const char* kScript = "C:\\projects\\mgexe\\MGE-XE\\mgeHost64\\shaders\\watch_shaders.py";
		if (GetFileAttributesA(kPython) == INVALID_FILE_ATTRIBUTES ||
			GetFileAttributesA(kScript) == INVALID_FILE_ATTRIBUTES) {
			return;   // not a dev tree — no watcher
		}
		// Gate (2): only the DEV install (the one watch_shaders.py DEPLOYs to) runs the watcher.
		{
			char exePath[MAX_PATH];
			DWORD n = GetModuleFileNameA(NULL, exePath, MAX_PATH);
			if (n == 0 || n >= MAX_PATH) { return; }
			for (DWORD i = 0; i <= n; ++i) {
				const char c = exePath[i];
				exePath[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;   // lowercase in place
			}
			if (!std::strstr(exePath, "\\mgem\\morrowind64\\")) {
				return;   // release / other install → no watcher, no window
			}
		}
		// Job object so the watcher dies with us even on a hard exit (~Client may not run): the OS
		// kills every process in the job when the last handle to it closes, i.e. when WE terminate.
		m_watcherJob = CreateJobObjectA(NULL, NULL);
		if (m_watcherJob) {
			JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = {};
			jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			SetInformationJobObject(m_watcherJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
		}
		char cmd[768];
		std::sprintf(cmd, "\"%s\" \"%s\"", kPython, kScript);
		STARTUPINFOA si = {}; si.cb = sizeof(si);
		// Start the console MINIMIZED and WITHOUT activating it, so it never steals focus from the game
		// or sits on top of it (CREATE_NEW_CONSOLE alone made a normal foreground window).
		si.dwFlags     = STARTF_USESHOWWINDOW;
		si.wShowWindow = SW_SHOWMINNOACTIVE;
		PROCESS_INFORMATION pi = {};
		// CREATE_NEW_CONSOLE keeps the watcher's own log window (minimized, per si above). Suspended so
		// we can assign it to the job BEFORE it runs.
		if (CreateProcessA(kPython, cmd, NULL, NULL, FALSE, CREATE_NEW_CONSOLE | CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
			if (m_watcherJob) { AssignProcessToJobObject(m_watcherJob, pi.hProcess); }
			ResumeThread(pi.hThread);
			m_watcherProcess = pi.hProcess;
			CloseHandle(pi.hThread);
			LOG::logline("FSL hot-reload watcher started (PID %u, minimized)", pi.dwProcessId);
		} else {
			LOG::winerror("Failed to start FSL hot-reload watcher");
			if (m_watcherJob) { CloseHandle(m_watcherJob); m_watcherJob = NULL; }
		}
	}

	void Client::stopWatcher() {
		if (m_watcherProcess != INVALID_HANDLE_VALUE) {
			TerminateProcess(m_watcherProcess, 0);
			CloseHandle(m_watcherProcess);
			m_watcherProcess = INVALID_HANDLE_VALUE;
		}
		if (m_watcherJob) {
			CloseHandle(m_watcherJob);   // KILL_ON_JOB_CLOSE finishes off the watcher if Terminate missed it
			m_watcherJob = NULL;
		}
	}

	bool Client::startServer(const char* executable) {
		STARTUPINFO startupInfo = {};
		PROCESS_INFORMATION processInfo = {};
		char strHandles[256] = { 0, };

		if (isServerActive()) {
			TerminateProcess(m_process, 0);
			CloseHandle(m_process);
			m_process = INVALID_HANDLE_VALUE;
			stopWatcher();   // kill the old watcher too; startWatcher() re-spawns below
		}

		CleanupHandle(m_sharedMem);
		CleanupHandle(m_rpcStartEvent);
		CleanupHandle(m_rpcCompleteEvent);

		// make the mapping handle inheritable
		SECURITY_ATTRIBUTES attrsAllowInherit = {
			sizeof(SECURITY_ATTRIBUTES), NULL, TRUE
		};
		m_sharedMem = CreateFileMappingA(INVALID_HANDLE_VALUE, &attrsAllowInherit, PAGE_READWRITE, 0, sizeof(Parameters), NULL);
		if (m_sharedMem == NULL) {
			LOG::winerror("Failed to create shared memory region");
			goto failedOnCreateMapping;
		}

		m_ipcParameters = static_cast<Parameters*>(MapViewOfFile(m_sharedMem, FILE_MAP_ALL_ACCESS, 0, 0, 0));
		if (m_ipcParameters == nullptr) {
			LOG::winerror("Failed to map shared memory region");
			goto failedOnMap;
		}

		ZeroMemory(m_ipcParameters, sizeof(Parameters));

		// get other handles the server will need
		HANDLE thisProcess;
		if (!DuplicateHandle(GetCurrentProcess(), GetCurrentProcess(), GetCurrentProcess(), &thisProcess, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
			LOG::winerror("Failed to duplicate current process handle");
			goto failedOnProcessHandle;
		}

		m_rpcStartEvent = CreateEventA(&attrsAllowInherit, FALSE, FALSE, NULL);
		if (m_rpcStartEvent == NULL) {
			LOG::winerror("Failed to create RPC start event");
			goto failedOnCreateEvent;
		}

		m_rpcCompleteEvent = CreateEventA(&attrsAllowInherit, FALSE, FALSE, NULL);
		if (m_rpcCompleteEvent == NULL) {
			LOG::winerror("Failed to create RPC complete event");
			goto failedOnCreateCompleteEvent;
		}

		// --- Dedicated geometry channel: second Parameters region + start/complete events ---
		m_geomSharedMem = CreateFileMappingA(INVALID_HANDLE_VALUE, &attrsAllowInherit, PAGE_READWRITE, 0, sizeof(Parameters), NULL);
		if (m_geomSharedMem == NULL) {
			LOG::winerror("Failed to create geometry shared memory region");
			goto failedOnGeomMapping;
		}
		m_geomParameters = static_cast<Parameters*>(MapViewOfFile(m_geomSharedMem, FILE_MAP_ALL_ACCESS, 0, 0, 0));
		if (m_geomParameters == nullptr) {
			LOG::winerror("Failed to map geometry shared memory region");
			goto failedOnGeomMap;
		}
		ZeroMemory(m_geomParameters, sizeof(Parameters));
		m_geomRpcStartEvent = CreateEventA(&attrsAllowInherit, FALSE, FALSE, NULL);
		if (m_geomRpcStartEvent == NULL) {
			LOG::winerror("Failed to create geometry RPC start event");
			goto failedOnGeomStartEvent;
		}
		m_geomRpcCompleteEvent = CreateEventA(&attrsAllowInherit, FALSE, FALSE, NULL);
		if (m_geomRpcCompleteEvent == NULL) {
			LOG::winerror("Failed to create geometry RPC complete event");
			goto failedOnGeomCompleteEvent;
		}

		std::sprintf(strHandles, "%p %p %p %p %p %p %p", m_sharedMem, thisProcess, m_rpcStartEvent, m_rpcCompleteEvent,
			m_geomSharedMem, m_geomRpcStartEvent, m_geomRpcCompleteEvent);
		if (!CreateProcessA(executable, strHandles, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, NULL, &startupInfo, &processInfo)) {
			LOG::winerror("Failed to start 64-bit host process %s", executable);
			goto failedOnCreateProcess;
		}

		m_process = processInfo.hProcess;
		// don't care about thread handle
		CloseHandle(processInfo.hThread);
		m_geomWaitHandles[0] = m_process;
		m_geomWaitHandles[1] = m_geomRpcCompleteEvent;

		LOG::logline("64-bit host process started (PID %u)", processInfo.dwProcessId);

		startWatcher();   // dev: FSL hot-reload watcher rides the host's lifecycle (no-op in shipped tree)

		// wait for the server to finish bootstrapping
		if (waitForCompletion() == WakeReason::Complete) {
			return true;
		}

		LOG::logline("Failed waiting for 64-bit host process to initialize");

	failedOnCreateProcess:
		CleanupHandle(m_geomRpcCompleteEvent);
	failedOnGeomCompleteEvent:
		CleanupHandle(m_geomRpcStartEvent);
	failedOnGeomStartEvent:
		UnmapViewOfFile(m_geomParameters);
		m_geomParameters = nullptr;
	failedOnGeomMap:
		CloseHandle(m_geomSharedMem);
		m_geomSharedMem = INVALID_HANDLE_VALUE;
	failedOnGeomMapping:
		CleanupHandle(m_rpcCompleteEvent);
	failedOnCreateCompleteEvent:
		CleanupHandle(m_rpcStartEvent);
	failedOnCreateEvent:
		CloseHandle(thisProcess);
	failedOnProcessHandle:
		UnmapViewOfFile(m_ipcParameters);
		m_ipcParameters = nullptr;
	failedOnMap:
		CloseHandle(m_sharedMem);
	failedOnCreateMapping:
		m_sharedMem = INVALID_HANDLE_VALUE;
		return false;
	}

	bool Client::beginRpc(Command command) {
		if (m_isRpcPending) {
			LOG::logline("Attempted RPC while another RPC was still in progress");
			return false;
		}

		// clear any unprocessed completion events
		ResetEvent(m_rpcCompleteEvent);

		m_ipcParameters->command = command;
		if (!SetEvent(m_rpcStartEvent)) {
			LOG::winerror("Failed to set RPC start event");
			return false;
		}

		m_isRpcPending = true;
		return true;
	}

	bool Client::allocVec(std::size_t elementSize, std::size_t windowSizeInElements, std::size_t maxSizeInElements, std::size_t initialCapacity) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.allocVecParams;
		params.elementSize = elementSize;
		params.windowSizeInElements = windowSizeInElements;
		params.maxCapacityInElements = maxSizeInElements;
		params.initialCapacity = initialCapacity;
		return beginRpc(Command::AllocVec);
	}

	bool Client::freeVec(VecId id) {
		WAIT_FOR_PREVIOUS_COMMAND;

		m_ipcParameters->params.freeVecParams.id = id;
		return beginRpc(Command::FreeVec);
	}

	bool Client::awaitFreeVec() {
		assert(m_ipcParameters->command == Command::FreeVec);

		if (waitForCompletion() != WakeReason::Complete) {
			LOG::logline("Vec free RPC failed");
			return false;
		}

		return m_ipcParameters->params.freeVecParams.wasFreed;
	}

	bool Client::freeVecBlocking(VecId id) {
		if (!freeVec(id)) {
			return false;
		}

		return awaitFreeVec();
	}

	bool Client::updateDynVis(VecId id) {
		WAIT_FOR_PREVIOUS_COMMAND;

		m_ipcParameters->params.dynVisParams.id = id;
		return beginRpc(Command::UpdateDynVis);
	}

	bool Client::initDistantStatics(VecId distantStatics, VecId distantSubsets) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.distantStaticParams;
		params.distantStatics = distantStatics;
		params.distantSubsets = distantSubsets;
		return beginRpc(Command::InitDistantStatics);
	}

	bool Client::initLandscape(VecId landscapeBuffers) {
		WAIT_FOR_PREVIOUS_COMMAND;

		m_ipcParameters->params.initLandscapeParams.buffers = landscapeBuffers;
		return beginRpc(Command::InitLandscape);
	}

	bool Client::setWorldSpaceBlocking(const std::string& cellname) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.worldSpaceParams;
		strncpy(params.cellname, cellname.c_str(), sizeof(params.cellname));
		params.cellname[sizeof(params.cellname) - 1] = 0;
		if (!beginRpc(Command::SetWorldSpace)) {
			return false;
		}

		if (waitForCompletion() != WakeReason::Complete) {
			return false;
		}

		return params.cellFound;
	}

	bool Client::getVisibleMeshesCoarse(VecId visibleSet, const ViewFrustum& viewFrustum, DWORD setFlags, VisibleSetSort sort) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.meshParams;
		params.visibleSet = visibleSet;
		params.viewFrustum = viewFrustum;
		params.sort = sort;
		params.setFlags = setFlags;
		return beginRpc(Command::GetVisibleMeshesCoarse);
	}

	bool Client::getVisibleMeshes(VecId visibleSet, const ViewFrustum& viewFrustum, const D3DXVECTOR4& viewSphere, DWORD setFlags, VisibleSetSort sort) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.meshParams;
		params.visibleSet = visibleSet;
		params.viewFrustum = viewFrustum;
		params.viewSphere = viewSphere;
		params.sort = sort;
		params.setFlags = setFlags;
		return beginRpc(Command::GetVisibleMeshes);
	}

	bool Client::getVisibleMeshesAllRanges(VecId visibleSet, std::uint8_t rangeCount,
		const ViewFrustum (&frustums)[3], const D3DXVECTOR4 (&spheres)[3],
		const DWORD (&setFlags)[3], VisibleSetSort sort,
		VecId reflSet, DWORD reflFlags,
		const ViewFrustum* reflFrustum, const D3DXVECTOR4* reflSphere,
		VisibleSetSort reflSort, VecId occlusionMask)
	{
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.meshAllRangesParams;
		params.visibleSet = visibleSet;
		params.sort = sort;
		params.rangeCount = rangeCount;
		for (std::uint8_t i = 0; i < 3; ++i) {
			params.viewFrustum[i] = frustums[i];
			params.viewSphere[i] = spheres[i];
			params.setFlags[i] = setFlags[i];
		}
		// Optional piggybacked reflection query. reflFlags=0 ⇒ skipped server-side.
		params.reflSet = reflSet;
		params.reflFlags = reflFlags;
		params.reflSort = reflSort;
		if (reflFlags != 0) {
			params.reflFrustum = *reflFrustum;
			params.reflSphere = *reflSphere;
		}
		// Host-side occlusion cull mask Vec (InvalidVector ⇒ disabled).
		params.occlusionMask = occlusionMask;
		return beginRpc(Command::GetVisibleMeshesAllRanges);
	}

	bool Client::sortVisibleSet(VecId visibleSet, VisibleSetSort sort) {
		WAIT_FOR_PREVIOUS_COMMAND;

		if (sort == VisibleSetSort::None) {
			SetEvent(m_rpcCompleteEvent);
			return true;
		}

		auto& params = m_ipcParameters->params.meshParams;
		params.visibleSet = visibleSet;
		params.sort = sort;
		return beginRpc(Command::SortVisibleSet);
	}

	bool Client::renderInitBlocking(std::uint32_t width, std::uint32_t height, std::uint32_t sampleCount,
		std::uint32_t anisoLevel, HANDLE sharedTexture0, HANDLE sharedTexture1, HANDLE* outFramebufferHandle) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.renderInitParams;
		params.width = width;
		params.height = height;
		params.sampleCount = sampleCount;
		params.anisoLevel = anisoLevel;
#pragma warning(push)
#pragma warning(disable: 4244 4302 4311)
		params.sharedTextureHandles[0] = static_cast<HANDLE32>(sharedTexture0);
		params.sharedTextureHandles[1] = static_cast<HANDLE32>(sharedTexture1);
#pragma warning(pop)
		params.framebufferHandle = nullptr;
		params.ok = false;
		if (!beginRpc(Command::RenderInit)) {
			return false;
		}

		if (waitForCompletion() != WakeReason::Complete) {
			return false;
		}

		if (params.ok && outFramebufferHandle) {
			*outFramebufferHandle = static_cast<HANDLE>(params.framebufferHandle);
		}
		return params.ok;
	}

	bool Client::renderFrameBlocking(std::uint32_t frameIndex, std::uint32_t targetIndex, double* outRenderMs) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.renderFrameParams;
		params.frameIndex = frameIndex;
		params.targetIndex = targetIndex;
		params.drawList = InvalidVector;   // triangle path (no scene data)
		params.drawCount = 0;
		params.drawBytes = 0;
		params.skinnedList = InvalidVector;
		params.skinnedCount = 0;
		params.skinnedBytes = 0;
		params.multiMapList = InvalidVector;
		params.multiMapCount = 0;
		params.multiMapBytes = 0;
		params.alphaList = InvalidVector;
		params.alphaCount = 0;
		params.alphaBytes = 0;
		params.capturedAlpha = InvalidVector;
		params.capturedVertBytes = 0;
		params.capturedIdxBytes = 0;
		params.bytesWritten = 0;
		params.renderMs = 0.0;
		if (!beginRpc(Command::RenderFrame)) {
			return false;
		}

		if (waitForCompletion() != WakeReason::Complete) {
			return false;
		}

		if (outRenderMs) {
			*outRenderMs = params.renderMs;
		}
		return params.bytesWritten > 0;
	}

	bool Client::renderSceneKickoff(std::uint32_t frameIndex, const float* viewProj,
		const float* lighting,
		VecId drawList, std::uint32_t drawCount, std::uint32_t drawBytes,
		VecId skinnedList, std::uint32_t skinnedCount, std::uint32_t skinnedBytes,
		VecId multiMapList, std::uint32_t multiMapCount, std::uint32_t multiMapBytes,
		VecId lightList, std::uint32_t lightCount, std::uint32_t lightBytes,
		VecId skyList, std::uint32_t skyCount, std::uint32_t skyBytes,
		VecId alphaList, std::uint32_t alphaCount, std::uint32_t alphaBytes,
		VecId capturedAlpha, std::uint32_t capturedVertBytes, std::uint32_t capturedIdxBytes,
		std::uint32_t debugMode,
		const DevInput* devInput,
		const float* waterParams, std::uint32_t waterEnabled,
		const FPFrame* fp) {
		WAIT_FOR_PREVIOUS_COMMAND;

		auto& params = m_ipcParameters->params.renderFrameParams;
		params.frameIndex = frameIndex;
		params.targetIndex = 0;
		std::memcpy(params.viewProj, viewProj, 16 * sizeof(float));
		std::memcpy(params.lighting, lighting, sizeof(params.lighting));
		params.drawList = drawList;
		params.drawCount = drawCount;
		params.drawBytes = drawBytes;
		params.skinnedList = skinnedList;
		params.skinnedCount = skinnedCount;
		params.skinnedBytes = skinnedBytes;
		params.multiMapList = multiMapList;
		params.multiMapCount = multiMapCount;
		params.multiMapBytes = multiMapBytes;
		params.lightList = lightList;
		params.lightCount = lightCount;
		params.lightBytes = lightBytes;
		params.skyList = skyList;
		params.skyCount = skyCount;
		params.skyBytes = skyBytes;
		params.alphaList = alphaList;
		params.alphaCount = alphaCount;
		params.alphaBytes = alphaBytes;
		params.capturedAlpha = capturedAlpha;
		params.capturedVertBytes = capturedVertBytes;
		params.capturedIdxBytes = capturedIdxBytes;
		params.debugMode = debugMode;
		// WT1 Forge water: 12 per-frame surface params + the F7 enable gate (no geometry).
		if (waterParams) { std::memcpy(params.waterParams, waterParams, 12 * sizeof(float)); }
		else { std::memset(params.waterParams, 0, 12 * sizeof(float)); }
		params.waterEnabled = waterEnabled;
		// FP1a first-person: one optional bundle; null ⇒ fpEnabled=0, host skips the FP pass.
		if (fp) {
			std::memcpy(params.fpViewProj, fp->viewProj, 16 * sizeof(float));
			params.fpDrawList = fp->drawList;
			params.fpDrawCount = fp->drawCount;
			params.fpDrawBytes = fp->drawBytes;
			params.fpSkinnedList = fp->skinnedList;
			params.fpSkinnedCount = fp->skinnedCount;
			params.fpSkinnedBytes = fp->skinnedBytes;
			params.fpAlphaList = fp->alphaList;
			params.fpAlphaCount = fp->alphaCount;
			params.fpAlphaBytes = fp->alphaBytes;
			params.fpEnabled = (fp->drawCount + fp->skinnedCount + fp->alphaCount) > 0 ? 1u : 0u;
		} else {
			std::memset(params.fpViewProj, 0, 16 * sizeof(float));
			params.fpDrawList = InvalidVector;
			params.fpDrawCount = params.fpDrawBytes = 0;
			params.fpSkinnedList = InvalidVector;
			params.fpSkinnedCount = params.fpSkinnedBytes = 0;
			params.fpAlphaList = InvalidVector;
			params.fpAlphaCount = params.fpAlphaBytes = 0;
			params.fpEnabled = 0;
		}
		const DevInput di = devInput ? *devInput : DevInput{};
		params.devMouseX = di.x;
		params.devMouseY = di.y;
		params.devMouseButtons = di.buttons;
		params.devMouseWheel = di.wheel;
		params.devUiVisible = di.uiVisible;
		params.devReloadShaders = di.reloadShaders;
		params.devDistLightsToggle = di.distLightsToggle;
		params.devFrameAhead = di.frameAhead;
		params.devClientWaitMs = di.clientWaitMs;
		params.devClientDtMs = di.clientDtMs;
		params.devClientMwStartMs = di.clientMwStartMs;
		params.bytesWritten = 0;
		params.renderMs = 0.0;
		if (!beginRpc(Command::RenderFrame)) {
			return false;
		}

		// The host is now rendering frame N while we return to MW's own frame-N work.
		// The window guard stays up until renderSceneFinish drains the completion —
		// see WAIT_FOR_PREVIOUS_COMMAND.
		m_frameWindowOpen = true;
		return true;
	}

	bool Client::renderSceneFinish(double* outRenderMs) {
		// ALWAYS drop the window guard, even on failure — a stale guard would refuse
		// every subsequent RPC forever (fail-loud, not fail-dead).
		m_frameWindowOpen = false;

		if (waitForCompletion() != WakeReason::Complete) {
			return false;
		}

		auto& params = m_ipcParameters->params.renderFrameParams;
		if (outRenderMs) {
			*outRenderMs = params.renderMs;
		}
		return params.bytesWritten > 0;
	}

	bool Client::renderSceneBlocking(std::uint32_t frameIndex, const float* viewProj,
		const float* lighting,
		VecId drawList, std::uint32_t drawCount, std::uint32_t drawBytes,
		VecId skinnedList, std::uint32_t skinnedCount, std::uint32_t skinnedBytes,
		VecId multiMapList, std::uint32_t multiMapCount, std::uint32_t multiMapBytes,
		VecId lightList, std::uint32_t lightCount, std::uint32_t lightBytes,
		VecId skyList, std::uint32_t skyCount, std::uint32_t skyBytes,
		VecId alphaList, std::uint32_t alphaCount, std::uint32_t alphaBytes,
		VecId capturedAlpha, std::uint32_t capturedVertBytes, std::uint32_t capturedIdxBytes,
		std::uint32_t debugMode,
		const DevInput* devInput,
		const float* waterParams, std::uint32_t waterEnabled,
		const FPFrame* fp,
		double* outRenderMs) {
		// Fused kickoff+finish — the exact pre-split serial behaviour (A/B reference).
		if (!renderSceneKickoff(frameIndex, viewProj, lighting,
			drawList, drawCount, drawBytes,
			skinnedList, skinnedCount, skinnedBytes,
			multiMapList, multiMapCount, multiMapBytes,
			lightList, lightCount, lightBytes,
			skyList, skyCount, skyBytes,
			alphaList, alphaCount, alphaBytes,
			capturedAlpha, capturedVertBytes, capturedIdxBytes,
			debugMode, devInput, waterParams, waterEnabled, fp)) {
			return false;
		}

		return renderSceneFinish(outRenderMs);
	}

	bool Client::geomUploadBlocking(VecId blob, std::uint32_t partCount, std::uint32_t byteCount, std::uint32_t* outUploaded) {
		// Geometry rides its OWN channel — wait only for the previous GEOM RPC, never the
		// main cull/scene channel. This is the whole point: bulk uploads no longer contend
		// with the one-at-a-time cull RPCs (which starved this at present time → exterior
		// black). The host services this channel on the same single thread via WFMO, so it
		// can never race renderScene.
		//
		// It CAN however queue behind an async RenderFrame (single host service thread) —
		// a mid-window upload silently blocks until the whole host frame completes,
		// defeating the overlap. The async window must be IPC-free on BOTH channels.
		if (m_frameWindowOpen) {
			++m_windowRefusals;
			LOG::logline("!! IPC geom upload attempted inside the async RenderFrame window (would serialize behind the host frame)");
			return false;
		}
		if (m_geomRpcPending && waitGeomCompletion() != WakeReason::Complete) {
			return false;
		}

		auto& params = m_geomParameters->params.geomUploadParams;
		params.blob = blob;
		params.partCount = partCount;
		params.byteCount = byteCount;
		params.partsUploaded = 0;
		if (!beginGeomRpc(Command::GeomUpload)) {
			return false;
		}

		if (waitGeomCompletion() != WakeReason::Complete) {
			return false;
		}

		if (outUploaded) {
			*outUploaded = params.partsUploaded;
		}
		return params.partsUploaded == partCount;
	}

	bool Client::texUploadBlocking(VecId blob, std::uint32_t texCount, std::uint32_t byteCount, std::uint32_t* outUploaded) {
		// Textures ride the SAME dedicated geometry channel as geomUpload (bulk, off the cull
		// channel). Wait only for the previous geom-channel RPC. Same async-window rule as
		// geomUploadBlocking: IPC-free on both channels while a RenderFrame is in flight.
		if (m_frameWindowOpen) {
			++m_windowRefusals;
			LOG::logline("!! IPC tex upload attempted inside the async RenderFrame window (would serialize behind the host frame)");
			return false;
		}
		if (m_geomRpcPending && waitGeomCompletion() != WakeReason::Complete) {
			return false;
		}

		auto& params = m_geomParameters->params.texUploadParams;
		params.blob = blob;
		params.texCount = texCount;
		params.byteCount = byteCount;
		params.texturesUploaded = 0;
		if (!beginGeomRpc(Command::TexUpload)) {
			return false;
		}

		if (waitGeomCompletion() != WakeReason::Complete) {
			return false;
		}

		if (outUploaded) {
			*outUploaded = params.texturesUploaded;
		}
		return params.texturesUploaded == texCount;
	}

	WakeReason Client::waitForCompletion(DWORD ms) {
		auto result = WaitForMultipleObjects(2, m_waitHandles, FALSE, ms);
		switch (result) {
		case WAIT_FAILED:
			LOG::winerror("IPC client wait for RPC completion failed");
			return WakeReason::Error;
		case WAIT_TIMEOUT:
			return WakeReason::Timeout;
		default:
			auto handleIndex = result - WAIT_OBJECT_0;
			switch (handleIndex) {
			case 0:
				return WakeReason::ServerLost;
			case 1:
				m_isRpcPending = false;
				return WakeReason::Complete;
			default:
				return WakeReason::Error;
			}
		}
	}

	WakeReason Client::tryWaitForCompletion(DWORD ms) {
		if (m_isRpcPending) {
			return waitForCompletion(ms);
		}

		return WakeReason::Complete;
	}

	bool Client::beginGeomRpc(Command command) {
		if (m_geomRpcPending) {
			LOG::logline("Attempted geometry RPC while another geometry RPC was still in progress");
			return false;
		}
		ResetEvent(m_geomRpcCompleteEvent);
		m_geomParameters->command = command;
		if (!SetEvent(m_geomRpcStartEvent)) {
			LOG::winerror("Failed to set geometry RPC start event");
			return false;
		}
		m_geomRpcPending = true;
		return true;
	}

	WakeReason Client::waitGeomCompletion(DWORD ms) {
		auto result = WaitForMultipleObjects(2, m_geomWaitHandles, FALSE, ms);
		switch (result) {
		case WAIT_FAILED:
			LOG::winerror("IPC client wait for geometry RPC completion failed");
			return WakeReason::Error;
		case WAIT_TIMEOUT:
			return WakeReason::Timeout;
		default:
			auto handleIndex = result - WAIT_OBJECT_0;
			switch (handleIndex) {
			case 0:   // m_process signalled — host gone
				return WakeReason::ServerLost;
			case 1:   // geometry RPC complete
				m_geomRpcPending = false;
				return WakeReason::Complete;
			default:
				return WakeReason::Error;
			}
		}
	}
}