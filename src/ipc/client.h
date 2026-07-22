#pragma once

#include "support/winheader.h"
#include "ipc/bridge.h"
#include "ipc/view.h"

#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace IPC {
	/**
	* @class Client
	* @brief An interface to the 64-bit server for the 32-bit client.
	* 
	* Client provides an interface to RPC to the server process. The current main purpose of the server is to host the
	* distant land QuadTrees and search them for visible meshes, providing the visible set to the client. To support
	* this, the server provides an API to allocate shared vectors in which the visible sets (or other data) can be
	* stored and shared with the client.
	* 
	* Because rendering is done on the client side, D3D resources must be allocated on the client side. However, the
	* server's distant land structures contain references to these resources. The server therefore provides APIs for
	* the client to push resource information to the server during distant land initialization. The remainder of the
	* API deals with controlling settings related to visible distant meshes and searching the QuadTrees for currently
	* visible meshes.
	* 
	* The server process must be started by the client via the startServer method. This method blocks until the server
	* signals that it has finished initializing and is ready to accept RPCs.
	* 
	* To provide opportunities for parallelism, all RPC methods are asynchronous except the ones that contain the word
	* "blocking". For asynchronous calls that return a value, there will be a corresponding `await` method that will
	* block until the result is available and return it. For asynchronous calls that don't return a value, you can use
	* the generic `waitForCompletion` method to wait for the RPC to complete. The blocking methods will return the
	* result directly.
	* 
	* Only one RPC may be active at a time. If an attempt is made to start another RPC while a previous one is still
	* active, the new RPC will block until the previous RPC is complete. This means that it's not necessary to
	* explicitly wait for RPC completion if you don't care about the results. You can fire off an asynchronous call and
	* move on, and the next RPC request will block if necessary before starting.
	*/
	class Client {
		HANDLE m_sharedMem;
		HANDLE m_rpcStartEvent;
		union {
			struct {
				HANDLE m_process;
				HANDLE m_rpcCompleteEvent;
			};
			HANDLE m_waitHandles[2];
		};
		Parameters* m_ipcParameters;
		bool m_isRpcPending;
		// Async-frame window: true between renderSceneKickoff and renderSceneFinish.
		// While open, ANY other RPC (main OR geom channel) is refused loudly — it would
		// either steal the RenderFrame completion (main) or serialize behind the whole
		// host frame on the single host service thread (geom). See client.cpp.
		bool m_frameWindowOpen;
		// Cumulative count of RPCs refused because the window was open (either channel).
		// Must stay 0 — nonzero means an unaudited call site fired mid-window (surfaced
		// as refuse= in the client's [hb] heartbeat, critical for ForgeFrameAhead where
		// the window spans the whole MW frame).
		unsigned m_windowRefusals;

		// Live render-scale: the current internal render resolution stamped into every render
		// RPC (kickoff / blocking). 0 ⇒ render at the host allocation size (default). Set by the
		// seam whenever the panel render-scale slider changes; persists across frames.
		std::uint32_t m_renderWidth = 0;
		std::uint32_t m_renderHeight = 0;

		// Dev FSL hot-reload watcher: a Windows-python child (watch_shaders.py) launched alongside
		// the host in a dev tree, killed with it. INVALID when not spawned (shipped tree / no python).
		HANDLE m_watcherProcess;
		// Job object owning the watcher, with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: when THIS process
		// (Morrowind) dies — even on a hard exit where ~Client never runs — the OS closes the job
		// handle and kills the watcher. Belt-and-suspenders with stopWatcher's explicit Terminate.
		HANDLE m_watcherJob;

		// Dedicated GEOMETRY channel: a SECOND shared-mem Parameters + start/complete
		// events to the SAME host process. Bulk geometry uploads run here so they never
		// contend with the one-at-a-time cull/scene RPCs on the main channel — that
		// contention starved the upload flush at present time and left exteriors black
		// (draw list referenced slots whose geometry never shipped). One user only (the
		// present-thread flush), so no cross-thread races on this channel; the host
		// services both channels on a single thread (WFMO), so uploadGeometry never races
		// renderScene. Separate Parameters union ⇒ no interleaved-completion clobber.
		HANDLE m_geomSharedMem;
		HANDLE m_geomRpcStartEvent;
		HANDLE m_geomRpcCompleteEvent;
		HANDLE m_geomWaitHandles[2];   // {m_process, m_geomRpcCompleteEvent}
		Parameters* m_geomParameters;
		bool m_geomRpcPending;

		bool beginRpc(Command command);
		bool beginGeomRpc(Command command);
		WakeReason waitGeomCompletion(DWORD ms = MaxWait);
		void startWatcher();   // dev FSL hot-reload watcher; no-op outside a dev tree
		void stopWatcher();

	public:
		Client();
		~Client();
		Client(const Client&) = delete;
		Client& operator=(const Client&) = delete;

		bool startServer(const char* executable);
		bool isServerActive();

		// Async-window observability: total RPCs refused because the RenderFrame window
		// was open. Must read 0 in every run (see m_windowRefusals).
		unsigned windowRefusals() const { return m_windowRefusals; }

		/**
		* @brief Asynchronously allocate a shared vector.
		* @param elementSize The size in bytes of the type of element that will be stored in the vector.
		* @param windowSizeInElements The number of elements that will be visible in a single window of the client view.
		*                             The size of the window will be rounded up to a multiple of the system allocation granularity.
		* @param maxSizeInElements The maximum number of elements to reserve memory for in the vector. The size of the
		*                          reservation will be rounded up to a multiple of the system allocation granularity.
		* @param initialCapacity Number of elements to initially commit memory for.
		* @return Whether the RPC was issued successfully.
		*/
		bool allocVec(std::size_t elementSize, std::size_t windowSizeInElements, std::size_t maxSizeInElements, std::size_t initialCapacity);

		/**
		* @brief Await the result of a previous asynchronous vector allocation.
		* @return A VecView with a view into the vector on success, or std::nullopt on failure.
		*/
		template<typename T>
		std::optional<VecView<T>> awaitAllocVec() {
			assert(m_ipcParameters->command == Command::AllocVec);

			auto result = waitForCompletion();
			if (result != WakeReason::Complete) {
				LOG::logline("Vec allocation RPC failed");
				return std::nullopt;
			}

			auto& params = m_ipcParameters->params.allocVecParams;
			if (params.id == InvalidVector) {
				LOG::logline("Vec allocation rejected by server");
				return std::nullopt;
			}

			assert(sizeof(T) == params.elementSize);

			// recalculate elements per window
			auto windowElements = params.windowBytes / sizeof(T);
			VecView<T> view(params.id, static_cast<VecBase::VecShare*>(params.header32), static_cast<std::size_t>(windowElements), static_cast<std::size_t>(params.windowBytes),
				static_cast<std::size_t>((params.reservedBytes / params.windowBytes) * windowElements), static_cast<std::size_t>(params.reservedBytes), static_cast<std::size_t>(params.headerBytes));
			if (!view.init()) {
				return std::nullopt;
			}

			return view;
		}

		/**
		* @brief Synchronously allocate a shared vector.
		* @param windowSizeInElements The number of elements that will be visible in a single window of the client view.
		*                             The size of the window will be rounded up to a multiple of the system allocation granularity.
		* @param maxSizeInElements The maximum number of elements to reserve memory for in the vector. The size of the
		*                          reservation will be rounded up to a multiple of the system allocation granularity.
		* @param initialCapacity Number of elements to initially commit memory for.
		* @return A VecView with a view into the vector on success, or std::nullopt on failure.
		*/
		template<typename T>
		std::optional<VecView<T>> allocVecBlocking(std::size_t windowSizeInElements, std::size_t maxSizeInElements, std::size_t initialCapacity) {
			if (!allocVec(sizeof(T), windowSizeInElements, maxSizeInElements, initialCapacity)) {
				return std::nullopt;
			}

			return awaitAllocVec<T>();
		}

		/**
		* @brief Asynchronously deallocate a shared vector.
		* 
		* Make sure any VecViews of the vector are destroyed prior to this call, otherwise the deallocation will fail.
		* 
		* @param id The ID of the vector to deallocate.
		* @return Whether the RPC was issued successfully.
		*/
		bool freeVec(VecId id);

		/**
		* @brief Await the result of a previous vector deallocation request.
		* @return Whether the vector was actually freed.
		*/
		bool awaitFreeVec();

		/**
		* @brief Synchronously deallocate a shared vector.
		*
		* Make sure any VecViews of the vector are destroyed prior to this call, otherwise the deallocation will fail.
		*
		* @param id The ID of the vector to deallocate.
		* @return Whether the vector was actually freed.
		*/
		bool freeVecBlocking(VecId id);

		/**
		* @brief Asynchronously update mesh dynamic visibility flags.
		* @param flags ID of a shared vector which the client has filled with flags to be updated.
		* @return Whether the RPC was issued successfully.
		*/
		bool updateDynVis(VecId flags);

		/**
		* @brief Inform the server of distant static D3D resources.
		* @param distantStatics ID of a shared vector of DistantStatic objects.
		* @param distantSubsets ID of a shared vector of DistantSubset objects.
		* @return Whether the RPC was issued successfully.
		*/
		bool initDistantStatics(VecId distantStatics, VecId distantSubsets);

		/**
		* @brief Inform the server of distant landscape D3D resources.
		* @param landscapeBuffers ID of a shared vector of LandscapeBuffers objects. The server expects to process this
		*                         vector in parallel with the client, so the client must use start_write/end_write and
		*                         begin filling the vector after calling this method.
		* @return Whether the RPC was issued successfully.
		*/
		bool initLandscape(VecId landscapeBuffers);

		/**
		* @brief Update the current worldspace by informing the server of the player's current cell.
		* @param cellname The name of the player's current cell.
		* @return Whether the RPC was successful.
		*/
		bool setWorldSpaceBlocking(const std::string& cellname);

		/**
		* @brief Asynchronously do a coarse search for visible meshes.
		* @param visibleSet ID of a shared vector of RenderMesh objects which will be populated with the results of the search.
		* @param viewFrustum The camera's current view frustum.
		* @param setFlags Flags indicating which types of meshes to search for. One or more of VIS_NEAR, VIS_FAR, VIS_VERY_FAR,
		*                 VIS_STATIC (= all 3 of the preceding flags), VIS_GRASS, or VIS_LAND.
		* @param sort The desired sorting of the result set, if any. If no sorting is requested (the default), the server will
		*             enable parallel writing on the vector, allowing the client to iterate over the results as they're
		*             populated, if desired (@ref VecView::start_read).
		* @return Whether the RPC was issued successfully.
		*/
		bool getVisibleMeshesCoarse(VecId visibleSet, const ViewFrustum& viewFrustum, DWORD setFlags, VisibleSetSort sort = VisibleSetSort::None);

		/**
		* @brief Asynchronously search for visible meshes.
		* @param visibleSet ID of a shared vector of RenderMesh objects which will be populated with the results of the search.
		* @param viewFrustum The camera's current view frustum.
		* @param viewSphere A sphere defining the region within the draw distance.
		* @param setFlags Flags indicating which types of meshes to search for. One or more of VIS_NEAR, VIS_FAR, VIS_VERY_FAR,
		*                 VIS_STATIC (= all 3 of the preceding flags), VIS_GRASS, or VIS_LAND.
		* @param sort The desired sorting of the result set, if any. If no sorting is requested (the default), the server will
		*             enable parallel writing on the vector, allowing the client to iterate over the results as they're
		*             populated, if desired (@ref VecView::start_read).
		* @return Whether the RPC was issued successfully.
		*/
		bool getVisibleMeshes(VecId visibleSet, const ViewFrustum& viewFrustum, const D3DXVECTOR4& viewSphere, DWORD setFlags, VisibleSetSort sort = VisibleSetSort::None);

		/**
		* @brief Batched 3-range variant of getVisibleMeshes. Runs all
		*        active range queries plus the sort in a single RPC so
		*        the server-side work can overlap with main-thread work
		*        between kick-off and the matching waitForCompletion.
		* @param rangeCount Number of active entries in the arrays (1..3).
		*                   Entries with setFlags=0 are skipped server-side.
		* @param reflSet Output vec for an optional piggybacked reflection-statics
		*                query, run independently of the 3 statics ranges into its
		*                own vec. reflFlags=0 (the default) disables it, leaving
		*                existing callers unchanged.
		*/
		bool getVisibleMeshesAllRanges(VecId visibleSet, std::uint8_t rangeCount,
			const ViewFrustum (&frustums)[3], const D3DXVECTOR4 (&spheres)[3],
			const DWORD (&setFlags)[3], VisibleSetSort sort,
			VecId reflSet = InvalidVector, DWORD reflFlags = 0,
			const ViewFrustum* reflFrustum = nullptr,
			const D3DXVECTOR4* reflSphere = nullptr,
			VisibleSetSort reflSort = VisibleSetSort::None,
			VecId occlusionMask = InvalidVector);

		/**
		* @brief Asynchronously sort an already-populated visible set.
		* @param visibleSet ID of a filled shared vector of RenderMesh objects.
		* @param sort The type of sort desired.
		* @return Whether the RPC was issued successfully.
		*/
		bool sortVisibleSet(VecId visibleSet, VisibleSetSort sort);

		/**
		* @brief Present-seam spike: initialize the host-side renderer.
		* @param width Target width in pixels.
		* @param height Target height in pixels.
		* @param sharedTexture0 D3D9Ex shared RT handle for zero-copy (Milestone B); null ⇒ CPU readback (A).
		* @param sharedTexture1 Second shared RT handle to double-buffer (Milestone C); null ⇒ single-buffered.
		* @param outFramebufferHandle A path only: receives a file-mapping HANDLE for the W*H*4 pixel blob (null on B).
		* @return Whether the host renderer initialized successfully (blocking).
		*/
		bool renderInitBlocking(std::uint32_t width, std::uint32_t height, std::uint32_t sampleCount,
			std::uint32_t anisoLevel, HANDLE sharedTexture0, HANDLE sharedTexture1, HANDLE* outFramebufferHandle);

		/**
		* @brief Present-seam spike: render one frame into shared buffer targetIndex.
		* @param frameIndex Frame counter (for logging only).
		* @param targetIndex Which shared buffer to render into (0/1 for double-buffering).
		* @param outRenderMs Optional out-param: host-side render time in ms.
		* @return Whether the frame was rendered successfully (blocking).
		*/
		bool renderFrameBlocking(std::uint32_t frameIndex, std::uint32_t targetIndex, double* outRenderMs = nullptr);

		/**
		* @brief M1c: render the cached opaque scene into the shared RT.
		* @param frameIndex Frame counter (logging).
		* @param viewProj 16 floats (D3DXMATRIX bytes, row-major) — the camera view*proj.
		* @param drawList Byte VecId of DrawItemWire[] (slot + world[16]); InvalidVector ⇒ triangle.
		* @param drawCount Number of draw items.
		* @param drawBytes Total bytes used in drawList.
		* @param outRenderMs Optional out: host render time (ms).
		* @return Whether the frame rendered (blocking).
		*/
		bool renderSceneBlocking(std::uint32_t frameIndex, const float* viewProj,
			const float* lighting,
			VecId drawList, std::uint32_t drawCount, std::uint32_t drawBytes,
			VecId skinnedList, std::uint32_t skinnedCount, std::uint32_t skinnedBytes,
			VecId multiMapList, std::uint32_t multiMapCount, std::uint32_t multiMapBytes,
			VecId lightList, std::uint32_t lightCount, std::uint32_t lightBytes,
			VecId skyList, std::uint32_t skyCount, std::uint32_t skyBytes,
			VecId alphaList = InvalidVector, std::uint32_t alphaCount = 0, std::uint32_t alphaBytes = 0,
			VecId capturedAlpha = InvalidVector, std::uint32_t capturedVertBytes = 0, std::uint32_t capturedIdxBytes = 0,
			std::uint32_t debugMode = 0,
			const DevInput* devInput = nullptr,
			const float* waterParams = nullptr, std::uint32_t waterEnabled = 0,
			const FPFrame* fp = nullptr,
			double* outRenderMs = nullptr);

		/**
		* @brief Async-frame split: copy the frame params into shared memory and start the
		*        host RenderFrame WITHOUT waiting. The host renders frame N while the caller
		*        continues MW's own frame-N work. Opens the IPC-free window: until the paired
		*        renderSceneFinish, every other RPC on either channel is refused loudly.
		*        Same arguments as renderSceneBlocking (minus outRenderMs, which the finish
		*        returns). All pointer args are copied before return — no lifetime coupling.
		* @return Whether the RPC was issued (host running). On false, no window is open.
		*/
		bool renderSceneKickoff(std::uint32_t frameIndex, const float* viewProj,
			const float* lighting,
			VecId drawList, std::uint32_t drawCount, std::uint32_t drawBytes,
			VecId skinnedList, std::uint32_t skinnedCount, std::uint32_t skinnedBytes,
			VecId multiMapList, std::uint32_t multiMapCount, std::uint32_t multiMapBytes,
			VecId lightList, std::uint32_t lightCount, std::uint32_t lightBytes,
			VecId skyList, std::uint32_t skyCount, std::uint32_t skyBytes,
			VecId alphaList = InvalidVector, std::uint32_t alphaCount = 0, std::uint32_t alphaBytes = 0,
			VecId capturedAlpha = InvalidVector, std::uint32_t capturedVertBytes = 0, std::uint32_t capturedIdxBytes = 0,
			std::uint32_t debugMode = 0,
			const DevInput* devInput = nullptr,
			const float* waterParams = nullptr, std::uint32_t waterEnabled = 0,
			const FPFrame* fp = nullptr);

		/**
		* @brief Async-frame split: wait for the RenderFrame started by renderSceneKickoff.
		*        The host fence-waits before replying, so on return the shared RT is
		*        GPU-complete and quiescent (safe to copy). Closes the IPC-free window
		*        unconditionally, even on failure.
		* @param outRenderMs Optional out: host-side render time in ms.
		* @return Whether the frame rendered (bytesWritten > 0).
		*/
		// outTimings (optional): the host's CPU/GPU phase split for the frame just drained, for
		// Tracy plots — see ipc/hostframetimings.h. Zeroed by the host on a failed frame.
		bool renderSceneFinish(double* outRenderMs = nullptr, HostFrameTimings* outTimings = nullptr);

		/**
		* @brief M1b: upload a batch of static opaque meshes to the Forge host.
		* @param blob A byte VecId filled with partCount packed parts (GeomPartWire+verts+indices).
		* @param partCount Number of parts packed in the blob.
		* @param byteCount Total bytes used in the blob.
		* @param outUploaded Optional out: parts the host actually built.
		* @return True if the host built every part (blocking).
		*/
		bool geomUploadBlocking(VecId blob, std::uint32_t partCount, std::uint32_t byteCount, std::uint32_t* outUploaded = nullptr);
		bool texUploadBlocking(VecId blob, std::uint32_t texCount, std::uint32_t byteCount, std::uint32_t* outUploaded = nullptr);

		// True if an RPC has been issued and not yet awaited. Use this to avoid
		// issuing a blocking RPC that would drain (steal) a pending async RPC's
		// completion — e.g. the geometry flush defers a frame rather than clobber
		// the shared Parameters union mid-pairing.
		bool isRpcPending() const { return m_isRpcPending; }

		// Live render-scale: stamp the current internal render resolution into every subsequent
		// render RPC. Persists until changed. 0,0 ⇒ host renders at its full allocation size.
		void setNextRenderSize(std::uint32_t w, std::uint32_t h) { m_renderWidth = w; m_renderHeight = h; }

		WakeReason waitForCompletion(DWORD ms = MaxWait);

		/**
		* @brief Wait for the current RPC to complete, but only if one is
		*        actually pending. Returns Complete immediately when no RPC
		*        is outstanding. Use this (rather than waitForCompletion) when
		*        an earlier interleaved RPC may already have drained the
		*        completion you were waiting for — otherwise waitForCompletion
		*        blocks on an event nobody will signal and times out at MaxWait.
		*/
		WakeReason tryWaitForCompletion(DWORD ms = MaxWait);
	};
}