#include "ipc/dlshare.h"
#include "ipc/server.h"
#include "support/log.h"

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
		// Sort runs once at the end across the merged set — pass None to
		// the per-range fetches so they don't sort intermediate state.
		LARGE_INTEGER t0, t;
		double rangeMs[3] = {0, 0, 0};
		QueryPerformanceCounter(&t0);
		LARGE_INTEGER prev = t0;
		for (std::uint8_t i = 0; i < params.rangeCount; ++i) {
			DistantLandShare::getVisibleMeshes(
				vec, params.viewFrustum[i], params.viewSphere[i],
				VisibleSetSort::None, params.setFlags[i]);
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

		g_allRangesStats.record(rangeMs, sortMs, walkMs + sortMs, vec.size());
	}

	void Server::sortVisibleSet() {
		auto& params = m_ipcParameters->params.meshParams;
		auto& vec = getVec<RenderMesh>(params.visibleSet);
		DistantLandShare::sortVisibleSet(vec, params.sort);
	}
}