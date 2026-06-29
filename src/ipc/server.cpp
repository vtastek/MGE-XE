#include "ipc/dlshare.h"
#include "ipc/server.h"
#include "ipc/occlusiontest.h"
#include "ipc/geomwire.h"
#include "support/log.h"
#include "vkrender.h"
#include "forgerender.h"

#include <cassert>
#include <cstring>

namespace {
    // --- getVisibleMeshesAllRanges profiling instrumentation ---
    // Breaks the AllRanges drain into per-range quadtree walks vs the final
    // merged sort, so we know which half of the ~3.5ms client-side wait to
    // attack. Rolling average logged to mgeHost64.log every kInterval calls
    // to avoid per-frame log spam.
    inline double msBetween(const LARGE_INTEGER& a, const LARGE_INTEGER& b) {
        static const double msPerTick = [] {
            LARGE_INTEGER f; QueryPerformanceFrequency(&f);
            return 1000.0 / static_cast<double>(f.QuadPart);
        }();
        return static_cast<double>(b.QuadPart - a.QuadPart) * msPerTick;
    }

    struct AllRangesStats {
        static constexpr int kInterval = 256;
        int samples = 0;
        double sumRange[3] = {0, 0, 0};
        double sumSort = 0;
        double sumTotal = 0;
        double maxTotal = 0;
        unsigned long long sumMeshes = 0;

        void record(const double range[3], double sortMs, double totalMs, unsigned meshes) {
            for (int i = 0; i < 3; ++i) sumRange[i] += range[i];
            sumSort += sortMs;
            sumTotal += totalMs;
            sumMeshes += meshes;
            if (totalMs > maxTotal) maxTotal = totalMs;
            if (++samples >= kInterval) {
                const double n = static_cast<double>(samples);
                LOG::logline("AllRanges avg/%d: r0=%.3f r1=%.3f r2=%.3f walk=%.3f sort=%.3f total=%.3f ms | maxTotal=%.3f | avgMeshes=%llu",
                    samples, sumRange[0] / n, sumRange[1] / n, sumRange[2] / n,
                    (sumRange[0] + sumRange[1] + sumRange[2]) / n, sumSort / n, sumTotal / n,
                    maxTotal, sumMeshes / static_cast<unsigned long long>(samples));
                LOG::flush();
                samples = 0;
                sumRange[0] = sumRange[1] = sumRange[2] = 0;
                sumSort = sumTotal = maxTotal = 0;
                sumMeshes = 0;
            }
        }
    };
    AllRangesStats g_allRangesStats;

    // Host-side occlusion mask, reconstructed from the shipped blob. Persistent
    // so the MOC instance survives across frames (recreated only on tier/res
    // change). The quadtree walk consults it via the OcclusionFilter callback
    // below, skipping occluded distant statics before PushBack so only visible
    // survivors cross the IPC boundary.
    OcclusionMask::HostMask g_hostMask;
    int g_hostMaskLogFrame   = 0;
    int g_hostCulledThisRpc  = 0;

    // --- Present-seam state ---
    // Target size of the Forge-owned shared render target (set at RenderInit), used
    // to report bytesWritten back to the client.
    std::uint32_t g_spikeWidth = 0;
    std::uint32_t g_spikeHeight = 0;

    // OcclusionFilter callback: raw per-frame verdict (matches MGE's in-process
    // cull — no inflate, no hysteresis, both unwired in MGE today). ctx is the
    // HostMask. Returns true to CULL (occluded). VIEW_CULLED / VISIBLE are kept
    // (the host's distant frustum is wider than the mask's). The QuadTreeMesh
    // pointer is available here, so a future host-side hysteresis streak map
    // would key on &m without any wire change.
    bool hostOcclCull(void* ctx, const QuadTreeMesh& m) {
        auto* mask = static_cast<OcclusionMask::HostMask*>(ctx);
        const auto r = mask->testSphere(
            m.sphere.center.x, m.sphere.center.y, m.sphere.center.z, m.sphere.radius);
        if (r == MaskedOcclusionCulling::OCCLUDED) {
            ++g_hostCulledThisRpc;
            return true;
        }
        return false;
    }
}

namespace IPC {
	Server::Server(HANDLE sharedMem, HANDLE clientProcess, HANDLE rpcStartEvent, HANDLE rpcCompleteEvent,
		HANDLE geomSharedMem, HANDLE geomRpcStartEvent, HANDLE geomRpcCompleteEvent) :
		m_sharedMem(sharedMem),
		m_clientProcess(clientProcess),
		m_rpcStartEvent(rpcStartEvent),
		m_rpcCompleteEvent(rpcCompleteEvent),
		m_geomSharedMem(geomSharedMem),
		m_geomRpcStartEvent(geomRpcStartEvent),
		m_geomRpcCompleteEvent(geomRpcCompleteEvent),
		m_geomParameters(nullptr),
		m_ipcParameters(nullptr),
		m_freeVecs()
	{ }

	Server::~Server() {
		// Tear down the Forge present-seam renderer if RenderInit brought it up.
		ForgeRender::shutdown();

		if (m_ipcParameters != nullptr) {
			UnmapViewOfFile(m_ipcParameters);
			m_ipcParameters = nullptr;
		}
		if (m_geomParameters != nullptr) {
			UnmapViewOfFile(m_geomParameters);
			m_geomParameters = nullptr;
		}

		CleanupHandle(m_sharedMem);
		CleanupHandle(m_clientProcess);
		CleanupHandle(m_rpcStartEvent);
		CleanupHandle(m_rpcCompleteEvent);
		CleanupHandle(m_geomSharedMem);
		CleanupHandle(m_geomRpcStartEvent);
		CleanupHandle(m_geomRpcCompleteEvent);
	}

	bool Server::complete() {
		if (!SetEvent(m_rpcCompleteEvent)) {
			LOG::winerror("Failed to signal RPC completion");
			return false;
		}

		return true;
	}

	bool Server::init() {
		if (m_ipcParameters != nullptr) {
			UnmapViewOfFile(m_ipcParameters);
			m_ipcParameters = nullptr;
		}

		m_ipcParameters = static_cast<Parameters*>(MapViewOfFile(m_sharedMem, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Parameters)));
		if (m_ipcParameters == nullptr) {
			LOG::winerror("Failed to map IPC parameters shared memory");
			return false;
		}

		// Map the dedicated geometry channel's Parameters, if the host was launched with one.
		if (m_geomSharedMem != nullptr) {
			if (m_geomParameters != nullptr) {
				UnmapViewOfFile(m_geomParameters);
				m_geomParameters = nullptr;
			}
			m_geomParameters = static_cast<Parameters*>(MapViewOfFile(m_geomSharedMem, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Parameters)));
			if (m_geomParameters == nullptr) {
				LOG::winerror("Failed to map geometry IPC parameters shared memory");
				return false;
			}
		}

		return true;
	}

	bool Server::listen() {
		// Init done — tell the client BOTH channels are ready to accept commands. (The
		// original single-channel loop signalled completion at the top of each iteration;
		// with two channels we signal per-RPC below and seed the init-complete here.)
		SetEvent(m_rpcCompleteEvent);
		if (m_geomRpcCompleteEvent != nullptr) {
			SetEvent(m_geomRpcCompleteEvent);
		}

		// 0 = client process, 1 = main RPC start, 2 = geometry RPC start (if present).
		// Single thread services both — so geometry upload (addResource into g_meshes)
		// can never race renderScene's draw, and the geometry channel just waits its turn
		// behind an in-flight cull/scene RPC instead of starving the present-time flush.
		HANDLE waits[3] = { m_clientProcess, m_rpcStartEvent, m_geomRpcStartEvent };
		const DWORD waitCount = (m_geomRpcStartEvent != nullptr) ? 3 : 2;

		while (true) {
			auto waitResult = WaitForMultipleObjects(waitCount, waits, FALSE, INFINITE);
			if (waitResult == WAIT_FAILED) {
				LOG::winerror("Failed to wait for RPC event");
				return false;
			}

			if (waitResult == WAIT_OBJECT_0) {
				LOG::logline("Morrowind process exited; exiting 64-bit host");
				return true;
			}

			if (waitResult == WAIT_OBJECT_0 + 2) {
				// Geometry channel — GeomUpload + TexUpload (both bulk, off the cull channel).
				if (m_geomParameters->command == Command::GeomUpload) {
					geomUpload();
				} else if (m_geomParameters->command == Command::TexUpload) {
					texUpload();
				} else if (m_geomParameters->command != Command::None) {
					LOG::logline("Geometry channel received unexpected command %u", m_geomParameters->command);
				}
				SetEvent(m_geomRpcCompleteEvent);
				continue;
			}

			// Main channel (WAIT_OBJECT_0 + 1).
			switch (m_ipcParameters->command) {
			case Command::None:
				break;
			case Command::AllocVec:
				allocVec();
				break;
			case Command::FreeVec:
				freeVec();
				break;
			case Command::Exit:
				LOG::logline("Host process received exit command");
				return true;
			case Command::UpdateDynVis:
				updateDynVis();
				break;
			case Command::InitDistantStatics:
				initDistantStatics();
				break;
			case Command::InitLandscape:
				initLandscape();
				break;
			case Command::SetWorldSpace:
				setWorldSpace();
				break;
			case Command::GetVisibleMeshesCoarse:
				getVisibleMeshesCoarse();
				break;
			case Command::GetVisibleMeshes:
				getVisibleMeshes();
				break;
			case Command::GetVisibleMeshesAllRanges:
				getVisibleMeshesAllRanges();
				break;
			case Command::SortVisibleSet:
				sortVisibleSet();
				break;
			case Command::RenderInit:
				renderInit();
				break;
			case Command::RenderFrame:
				renderFrame();
				break;
			case Command::GeomUpload:
				// Geometry now rides its own channel; a GeomUpload on the main channel is
				// unexpected (would read the wrong Parameters). Single-channel fallback only.
				if (m_geomParameters == nullptr) {
					geomUpload();
				} else {
					LOG::logline("Main channel received GeomUpload but a geometry channel exists");
				}
				break;
			default:
				LOG::logline("Received unknown command value %u", m_ipcParameters->command);
				break;
			}
			SetEvent(m_rpcCompleteEvent);
		}
	}

	template<typename T>
	Vec<T>& Server::getVec(VecId id) {
		auto pVec = m_vecs[id];
		// when the client requests to allocate a shared vector, there's no way we can communicate a template
		// argument to the server, so all vectors are stored with a dummy type of char. the actual contained
		// type doesn't affect the layout of the vector (as all it holds is a pointer to the elements), so
		// we can freely cast between types without breaking the class itself. we will do an assert to make
		// sure the size of the type we're being told the vector contains matches the size of the type it
		// was told it contained when it was created.
		assert(sizeof(T) == pVec->m_elementBytes);
		return *reinterpret_cast<Vec<T>*>(pVec);
	}

	bool Server::allocVec() {
		auto& params = m_ipcParameters->params.allocVecParams;

		Vec<char>* vec = nullptr;
		VecId id = InvalidVector;
		if (!m_freeVecs.empty()) {
			id = m_freeVecs.front();
			m_freeVecs.pop();
			m_vecs[id] = vec = new Vec<char>(id, nullptr, params.maxCapacityInElements, params.windowSizeInElements, params.elementSize);
		} else {
			id = static_cast<VecId>(m_vecs.size());
			vec = new Vec<char>(id, nullptr, params.maxCapacityInElements, params.windowSizeInElements, params.elementSize);
			m_vecs.push_back(vec);
		}

		m_ipcParameters->params.allocVecParams.id = id;

		if (!(vec->init(m_clientProcess, params) && vec->reserve(params.initialCapacity))) {
			delete vec;
			m_vecs[id] = nullptr;
			// mark this slot free again
			m_freeVecs.push(id);
			return false;
		}

		return true;
	}

	bool Server::freeVec() {
		auto& params = m_ipcParameters->params.freeVecParams;
		params.wasFreed = false;

		auto& vec = m_vecs[params.id];
		if (vec != nullptr) {
			if (!vec->can_free())
				return false;

			delete vec;
			vec = nullptr;
			m_freeVecs.push(params.id);
			params.wasFreed = true;
		}

		return true;
	}

	void Server::updateDynVis() {
		auto& params = m_ipcParameters->params.dynVisParams;
		auto& vec = getVec<DynVisFlag>(params.id);
		for (auto& update : vec) {
			for (auto mesh : DistantLandShare::dynamicVisGroupsServer[update.groupIndex]) {
				mesh->enabled = update.enable;
			}
		}
	}

	bool Server::initDistantStatics() {
		auto& params = m_ipcParameters->params.distantStaticParams;
		auto& distantStatics = getVec<DistantStatic>(params.distantStatics);
		auto& distantSubsets = getVec<DistantSubset>(params.distantSubsets);
		return DistantLandShare::initDistantStaticsServer(distantStatics, distantSubsets);
	}

	bool Server::initLandscape() {
		auto& params = m_ipcParameters->params.initLandscapeParams;
		return DistantLandShare::initLandscapeServer(getVec<LandscapeBuffers>(params.buffers), params.texWorldColour);
	}

	void Server::setWorldSpace() {
		auto& params = m_ipcParameters->params.worldSpaceParams;
		params.cellFound = DistantLandShare::setCurrentWorldSpace(params.cellname);
	}

	void Server::getVisibleMeshesCoarse() {
		auto& params = m_ipcParameters->params.meshParams;
		auto& vec = getVec<RenderMesh>(params.visibleSet);
		DistantLandShare::getVisibleMeshesCoarse(vec, params.viewFrustum, params.sort, params.setFlags);
	}

	void Server::getVisibleMeshes() {
		auto& params = m_ipcParameters->params.meshParams;
		auto& vec = getVec<RenderMesh>(params.visibleSet);
		DistantLandShare::getVisibleMeshes(vec, params.viewFrustum, params.viewSphere, params.sort, params.setFlags);
	}

	// Batched all-ranges variant. Runs each of the rangeCount range queries
	// (Near/Far/VeryFar) into the same vec, then applies the requested sort
	// once over the merged set. Saves 3 client-server round trips vs the
	// per-range RPC sequence.
	void Server::getVisibleMeshesAllRanges() {
		auto& params = m_ipcParameters->params.meshAllRangesParams;
		auto& vec = getVec<RenderMesh>(params.visibleSet);

		// Host-side occlusion cull: reconstruct the shipped mask and build the
		// filter the walk consults per-mesh (occluded → skipped pre-PushBack, so
		// only survivors cross IPC). Built BEFORE the walks. Any failure (no mask
		// shipped, stale snapshot, version/layout mismatch) leaves occPtr null →
		// the walk behaves exactly as before (full set).
		OcclusionFilter occ{};
		const OcclusionFilter* occPtr = nullptr;
		bool maskLoaded = false;
		std::uint32_t availBytes = 0;
		if (params.occlusionMask != InvalidVector) {
			auto& mv = getVec<OcclusionMask::MaskChunk>(params.occlusionMask);
			availBytes = mv.size() * static_cast<std::uint32_t>(sizeof(OcclusionMask::MaskChunk));
			if (mv.size() >= 1 && availBytes >= sizeof(OcclusionMask::Header)) {
				maskLoaded = g_hostMask.load(&mv[0], static_cast<int>(availBytes));
				if (maskLoaded) {
					occ.cull = &hostOcclCull;
					occ.ctx  = &g_hostMask;
					occPtr   = &occ;
				}
			}
		}
		g_hostCulledThisRpc = 0;

		// Sort runs once at the end across the merged set — pass None to
		// the per-range fetches so they don't sort intermediate state.
		LARGE_INTEGER t0, t;
		double rangeMs[3] = {0, 0, 0};
		QueryPerformanceCounter(&t0);
		LARGE_INTEGER prev = t0;
		for (std::uint8_t i = 0; i < params.rangeCount; ++i) {
			DistantLandShare::getVisibleMeshes(
				vec, params.viewFrustum[i], params.viewSphere[i],
				VisibleSetSort::None, params.setFlags[i], occPtr);
			QueryPerformanceCounter(&t);
			if (i < 3) rangeMs[i] = msBetween(prev, t);
			prev = t;  // prev now marks the end of the walks
		}
		const double walkMs = msBetween(t0, prev);
		double sortMs = 0;
		if (params.sort != VisibleSetSort::None) {
			DistantLandShare::sortVisibleSet(vec, params.sort);
			QueryPerformanceCounter(&t);
			sortMs = msBetween(prev, t);
		}

		// Optional piggybacked reflection-statics query into a separate vec.
		// Its server-cull overlaps the client's kickoff→drain head-start window
		// (GeomCache walk + sky), so the reflection set is ready by the time the
		// worker drains this single RPC. reflFlags==0 ⇒ no reflection this RPC.
		if (params.reflFlags != 0) {
			auto& rvec = getVec<RenderMesh>(params.reflSet);
			DistantLandShare::getVisibleMeshes(
				rvec, params.reflFrustum, params.reflSphere,
				VisibleSetSort::None, params.reflFlags);
			DistantLandShare::sortVisibleSet(rvec, params.reflSort);
		}

		// Host-side occlusion cull diagnostic (Increment 2). The walk already
		// dropped occluded statics; vec is now the survivor set. Log the cull
		// count + survivors every 60 frames when a mask was shipped.
		if (params.occlusionMask != InvalidVector && (g_hostMaskLogFrame++ % 60) == 0) {
			const auto& h = g_hostMask.header();
			LOG::logline("-- [host occl] avail=%u B %s | ready=%d impl=%d res=%dx%d zbuf=%u age=%llums "
			             "| culled=%d survivors=%u",
			             availBytes, maskLoaded ? "loaded" : "REJECTED",
			             h.ready, h.impl, h.maskW, h.maskH, h.zbufBytes, h.ageMs,
			             g_hostCulledThisRpc, vec.size());
			LOG::flush();
		}

		g_allRangesStats.record(rangeMs, sortMs, walkMs + sortMs, vec.size());
	}

	void Server::sortVisibleSet() {
		auto& params = m_ipcParameters->params.meshParams;
		auto& vec = getVec<RenderMesh>(params.visibleSet);
		DistantLandShare::sortVisibleSet(vec, params.sort);
	}

	// --- Present seam (route 1: Forge D3D12 host renderer) ---
	// The Forge host creates a SHARED D3D12 render target + NT handle; we duplicate
	// that handle into the client (MW) process, which ingests it via D3D9Ex
	// CreateTexture. (The Vulkan VKRender path stays in the tree as the CPU-staging
	// fallback but is no longer wired here — see project_d4_present_seam_wiring.)
	// MW's old B-path sharedTextureHandles[] are ignored: the host now OWNS the RT.
	void Server::renderInit() {
		auto& params = m_ipcParameters->params.renderInitParams;
		params.framebufferHandle = nullptr;   // reused as the shared-RT NT handle (client-process value)
		params.ok = false;

		if (!ForgeRender::init(params.width, params.height, params.sampleCount, params.anisoLevel)) {
			LOG::logline("!! [seam] ForgeRender::init(%ux%u, %ux MSAA, AF %u) failed", params.width, params.height, params.sampleCount, params.anisoLevel);
			return;
		}

		g_spikeWidth = params.width;
		g_spikeHeight = params.height;

		HANDLE hostHandle = static_cast<HANDLE>(ForgeRender::sharedHandle());
		if (hostHandle == nullptr) {
			LOG::logline("!! [seam] ForgeRender produced no shared handle");
			ForgeRender::shutdown();
			return;
		}

		// Duplicate the host's NT shared-RT handle into the client (MW) process, so
		// MW's D3D9Ex can ingest it directly as pSharedHandle (same cross-process
		// mechanism Vec::init / the old A-path used).
		HANDLE clientHandle = INVALID_HANDLE_VALUE;
		if (!DuplicateHandle(GetCurrentProcess(), hostHandle, m_clientProcess, &clientHandle,
				0, FALSE, DUPLICATE_SAME_ACCESS)) {
			LOG::winerror("[seam] failed to duplicate shared-RT handle to client");
			ForgeRender::shutdown();
			return;
		}

#pragma warning(push)
#pragma warning(disable: 4244 4302 4311)
		params.framebufferHandle = static_cast<HANDLE32>(clientHandle);
#pragma warning(pop)
		params.ok = true;
		LOG::logline(">> [seam] render init ok (%ux%u, Forge shared RT, host handle %p -> client %p) sceneReady=%d",
			params.width, params.height, hostHandle, clientHandle, (int)ForgeRender::sceneReady());
		LOG::flush();
	}

	// Render one frame into the shared RT. Blocking (host fence-waits), so the reply
	// implies the frame is GPU-complete and MW's StretchRect won't race the draw.
	void Server::renderFrame() {
		auto& params = m_ipcParameters->params.renderFrameParams;
		params.bytesWritten = 0;
		params.renderMs = 0.0;

		LARGE_INTEGER t0; QueryPerformanceCounter(&t0);
		bool ok;
		if (params.drawList != InvalidVector || params.skinnedList != InvalidVector
			|| params.multiMapList != InvalidVector || params.skyList != InvalidVector) {
			// M1c/M-Skinning scene path: static DrawItemWire[] (drawList) and/or skinned
			// [SkinnedDrawWire][palette]* (skinnedList) + inline camera. Either may be Invalid.
			const void* drawPtr = nullptr;
			std::uint32_t bytes = 0;
			if (params.drawList != InvalidVector) {
				auto& vec = getVec<IPC::GeomChunk>(params.drawList);
				const std::uint32_t availBytes = vec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
				bytes = params.drawBytes;
				if (bytes == 0 || bytes > availBytes) {
					bytes = availBytes;
				}
				drawPtr = vec.size() ? &vec[0] : nullptr;
			}
			const void* skinnedPtr = nullptr;
			std::uint32_t skinnedBytes = 0;
			if (params.skinnedList != InvalidVector) {
				auto& svec = getVec<IPC::GeomChunk>(params.skinnedList);
				const std::uint32_t savail = svec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
				skinnedBytes = params.skinnedBytes;
				if (skinnedBytes == 0 || skinnedBytes > savail) {
					skinnedBytes = savail;
				}
				skinnedPtr = svec.size() ? &svec[0] : nullptr;
			}
			const void* multiMapPtr = nullptr;
			std::uint32_t multiMapBytes = 0;
			if (params.multiMapList != InvalidVector) {
				auto& mvec = getVec<IPC::GeomChunk>(params.multiMapList);
				const std::uint32_t mavail = mvec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
				multiMapBytes = params.multiMapBytes;
				if (multiMapBytes == 0 || multiMapBytes > mavail) {
					multiMapBytes = mavail;
				}
				multiMapPtr = mvec.size() ? &mvec[0] : nullptr;
			}
			const void* lightPtr = nullptr;
			std::uint32_t lightBytes = 0;
			if (params.lightList != InvalidVector) {
				auto& lvec = getVec<IPC::GeomChunk>(params.lightList);
				const std::uint32_t lavail = lvec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
				lightBytes = params.lightBytes;
				if (lightBytes == 0 || lightBytes > lavail) {
					lightBytes = lavail;
				}
				lightPtr = lvec.size() ? &lvec[0] : nullptr;
			}
			const void* skyPtr = nullptr;
			std::uint32_t skyBytes = 0;
			if (params.skyList != InvalidVector) {
				auto& kvec = getVec<IPC::GeomChunk>(params.skyList);
				const std::uint32_t kavail = kvec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
				skyBytes = params.skyBytes;
				if (skyBytes == 0 || skyBytes > kavail) {
					skyBytes = kavail;
				}
				skyPtr = kvec.size() ? &kvec[0] : nullptr;
			}
			static unsigned s_sceneLog = 0;
			const bool logScene = (s_sceneLog++ % 60) == 0;
			if (logScene) {
				LOG::logline(">> [scene] renderScene ENTER frame=%u drawCount=%u bytes=%u skinnedCount=%u skinnedBytes=%u mmCount=%u lightCount=%u",
					params.frameIndex, params.drawCount, bytes, params.skinnedCount, skinnedBytes, params.multiMapCount, params.lightCount);
				LOG::flush();
			}
			ForgeRender::setDebugMode(params.debugMode);
			ForgeRender::setDevInput(params.devMouseX, params.devMouseY, params.devMouseButtons,
				params.devMouseWheel, params.devUiVisible);
			if (params.devReloadShaders) {
				ForgeRender::reloadComputeShaders();
			}
			ok = ForgeRender::renderScene(params.viewProj, params.lighting, drawPtr, params.drawCount, bytes,
				skinnedPtr, params.skinnedCount, skinnedBytes,
				multiMapPtr, params.multiMapCount, multiMapBytes,
				lightPtr, params.lightCount, lightBytes,
				skyPtr, params.skyCount, skyBytes);
			if (logScene) {
				LOG::logline(">> [scene] renderScene DONE ok=%d drawn=%u skinned=%u",
					(int)ok, ForgeRender::lastDrawn(), ForgeRender::lastSkinnedDrawn());
				LOG::flush();
			}
		} else {
			ok = ForgeRender::renderFrame(params.frameIndex);
		}
		if (!ok) {
			return;
		}
		LARGE_INTEGER t1; QueryPerformanceCounter(&t1);

		params.bytesWritten = g_spikeWidth * g_spikeHeight * 4u;
		params.renderMs = msBetween(t0, t1);
	}

	// M1b: hand the packed geometry blob to the Forge host, which builds a D3D12
	// VB/IB per part and stores it slot-indexed. The blob is single-window, so
	// &vec[0] is a contiguous pointer to the whole batch (as for the occlusion mask).
	void Server::geomUpload() {
		// Read from the geometry channel's Parameters when present (its own shared region);
		// fall back to the main channel only in the single-channel launch.
		Parameters* pp = (m_geomParameters != nullptr) ? m_geomParameters : m_ipcParameters;
		auto& params = pp->params.geomUploadParams;
		params.partsUploaded = 0;
		if (params.blob == InvalidVector || params.partCount == 0) {
			return;
		}
		auto& vec = getVec<IPC::GeomChunk>(params.blob);
		if (vec.size() == 0) {
			return;
		}
		const std::uint32_t availBytes = vec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
		std::uint32_t bytes = params.byteCount;
		if (bytes == 0 || bytes > availBytes) {
			bytes = availBytes;
		}
		const void* p = &vec[0];
		IPC::GeomPartWire h0 = {};
		if (bytes >= sizeof(h0)) {
			std::memcpy(&h0, p, sizeof(h0));
		}
		// Log BEFORE the call (flushed): if the matching "built" line below never
		// appears, uploadGeometry crashed/hung on this input. Only for multi-part batches
		// (cell-load bursts — the risky path). A single-part upload every frame is the
		// steady animated-mesh re-upload; logging it spammed the host log (partsUploaded
		// is 0 pre-call, so the old `!= partCount` test was always true for partCount=1).
		const bool logThis = (params.partCount > 1);
		if (logThis) {
			LOG::logline(">> [geom] geomUpload ENTER: partCount=%u size=%u availBytes=%u byteCount=%u bytes=%u p0{slot=%u rev=%u v=%u i=%u}",
				params.partCount, vec.size(), availBytes, params.byteCount, bytes,
				h0.slot, h0.revisionID, h0.vertexCount, h0.indexCount);
			LOG::flush();
		}
		params.partsUploaded = ForgeRender::uploadGeometry(p, bytes, params.partCount);
		if (params.partCount > 1 || params.partsUploaded != params.partCount) {
			LOG::logline(">> [geom] geomUpload DONE: built %u/%u", params.partsUploaded, params.partCount);
			LOG::flush();
		}
	}

	void Server::texUpload() {
		Parameters* pp = (m_geomParameters != nullptr) ? m_geomParameters : m_ipcParameters;
		auto& params = pp->params.texUploadParams;
		params.texturesUploaded = 0;
		if (params.blob == InvalidVector || params.texCount == 0) {
			return;
		}
		auto& vec = getVec<IPC::GeomChunk>(params.blob);
		if (vec.size() == 0) {
			return;
		}
		const std::uint32_t availBytes = vec.size() * static_cast<std::uint32_t>(sizeof(IPC::GeomChunk));
		std::uint32_t bytes = params.byteCount;
		if (bytes == 0 || bytes > availBytes) {
			bytes = availBytes;
		}
		params.texturesUploaded = ForgeRender::uploadTextures(&vec[0], bytes, params.texCount);
		LOG::logline(">> [tex] texUpload DONE: built %u/%u (%u bytes)",
			params.texturesUploaded, params.texCount, bytes);
		LOG::flush();
	}
}