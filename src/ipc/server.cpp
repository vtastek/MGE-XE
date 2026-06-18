#include "ipc/dlshare.h"
#include "ipc/server.h"
#include "ipc/occlusiontest.h"
#include "support/log.h"
#include "vkrender.h"

#include <cassert>

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

    // --- Present-seam spike state ---
    // Host-owned flat framebuffer mapping. g_spikeFbLocal is the 64-bit view the
    // Vulkan readback writes into; the duplicated 32-bit handle is returned to the
    // client so it can map the same blob for the composite blit.
    HANDLE g_spikeFbMap = nullptr;
    void* g_spikeFbLocal = nullptr;
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
	Server::Server(HANDLE sharedMem, HANDLE clientProcess, HANDLE rpcStartEvent, HANDLE rpcCompleteEvent) :
		m_sharedMem(sharedMem),
		m_clientProcess(clientProcess),
		m_rpcStartEvent(rpcStartEvent),
		m_rpcCompleteEvent(rpcCompleteEvent),
		m_ipcParameters(nullptr),
		m_freeVecs()
	{ }

	Server::~Server() {
		if (m_ipcParameters != nullptr) {
			UnmapViewOfFile(m_ipcParameters);
			m_ipcParameters = nullptr;
		}

		CleanupHandle(m_sharedMem);
		CleanupHandle(m_clientProcess);
		CleanupHandle(m_rpcStartEvent);
		CleanupHandle(m_rpcCompleteEvent);
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

		return true;
	}

	bool Server::listen() {
		while (true) {
			// signal the completion of whatever we were doing before (also signals that we've finished initializing on the first iteration)
			SetEvent(m_rpcCompleteEvent);

			// 0 = client process, 1 = RPC start event
			auto waitResult = WaitForMultipleObjects(2, m_waitHandles, FALSE, INFINITE);
			if (waitResult == WAIT_FAILED) {
				LOG::winerror("Failed to wait for RPC event");
				return false;
			}

			if (waitResult == WAIT_OBJECT_0) {
				LOG::logline("Morrowind process exited; exiting 64-bit host");
				return true;
			}

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
			default:
				LOG::logline("Received unknown command value %u", m_ipcParameters->command);
				break;
			}
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

	// --- Present-seam spike ---
	// Bring up the Vulkan offscreen renderer, create the flat framebuffer mapping,
	// and duplicate its handle into the client process for the composite blit.
	void Server::renderInit() {
		auto& params = m_ipcParameters->params.renderInitParams;
		params.framebufferHandle = nullptr;
		params.ok = false;

		if (!VKRender::init(params.width, params.height)) {
			LOG::logline("!! [spike] VKRender::init(%ux%u) failed", params.width, params.height);
			return;
		}

		// Tear down any prior mapping (re-init on size change).
		if (g_spikeFbLocal) { UnmapViewOfFile(g_spikeFbLocal); g_spikeFbLocal = nullptr; }
		if (g_spikeFbMap)   { CloseHandle(g_spikeFbMap); g_spikeFbMap = nullptr; }

		const std::uint32_t bytes = params.width * params.height * 4u;
		g_spikeFbMap = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, bytes, NULL);
		if (g_spikeFbMap == NULL) {
			LOG::winerror("[spike] failed to create framebuffer mapping (%u bytes)", bytes);
			g_spikeFbMap = nullptr;
			return;
		}
		g_spikeFbLocal = MapViewOfFile(g_spikeFbMap, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
		if (g_spikeFbLocal == nullptr) {
			LOG::winerror("[spike] failed to map framebuffer locally");
			CloseHandle(g_spikeFbMap);
			g_spikeFbMap = nullptr;
			return;
		}

		// Duplicate the mapping handle into the client process (same mechanism Vec::init uses).
		HANDLE clientHandle = INVALID_HANDLE_VALUE;
		if (!DuplicateHandle(GetCurrentProcess(), g_spikeFbMap, m_clientProcess, &clientHandle,
				0, FALSE, DUPLICATE_SAME_ACCESS)) {
			LOG::winerror("[spike] failed to duplicate framebuffer handle to client");
			UnmapViewOfFile(g_spikeFbLocal); g_spikeFbLocal = nullptr;
			CloseHandle(g_spikeFbMap); g_spikeFbMap = nullptr;
			return;
		}

		g_spikeWidth = params.width;
		g_spikeHeight = params.height;
#pragma warning(push)
#pragma warning(disable: 4244 4302 4311)
		params.framebufferHandle = static_cast<HANDLE32>(clientHandle);
#pragma warning(pop)
		params.ok = true;
		LOG::logline(">> [spike] render init ok (%ux%u, %u bytes, client handle %p)",
			params.width, params.height, bytes, clientHandle);
	}

	// Render one triangle frame and copy the pixels into the flat framebuffer mapping.
	void Server::renderFrame() {
		auto& params = m_ipcParameters->params.renderFrameParams;
		params.bytesWritten = 0;
		params.renderMs = 0.0;

		if (g_spikeFbLocal == nullptr || !VKRender::isReady()) {
			return;
		}

		const std::uint32_t bytes = g_spikeWidth * g_spikeHeight * 4u;
		double renderMs = 0.0;
		if (!VKRender::renderFrame(g_spikeFbLocal, bytes, &renderMs)) {
			return;
		}

		params.bytesWritten = bytes;
		params.renderMs = renderMs;
	}
}