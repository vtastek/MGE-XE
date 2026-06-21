#pragma once

#include "ipc/bridge.h"
#include "ipc/vec.h"
#include <queue>
#include <vector>

namespace IPC {
	/**
	* @class Server
	* @brief Listener for RPC commands to the 64-bit server process.
	* 
	* Server implements the various RPC commands that the client process may request. The server process is started
	* by the client process and exits automatically when the client process exits. The server process exists to host
	* distant land structures and provide an API to share distant land information with the client. The implementation
	* of this API is handled within the Server class itself via the `listen` method, which will listen for and execute
	* commands until the client process exits or the client instructs the server to exit.
	* 
	* Only one RPC may be active at a time. The client must wait for the completion signal before sending another RPC.
	* The client must also wait for the completion signal after starting the server process, which will be signaled
	* when the server is ready to begin accepting commands.
	*/
	class Server {
		HANDLE m_sharedMem;
		union {
			struct {
				HANDLE m_clientProcess;
				HANDLE m_rpcStartEvent;
			};
			HANDLE m_waitHandles[2];
		};
		HANDLE m_rpcCompleteEvent;
		// Dedicated geometry channel (second Parameters + start/complete events). Serviced
		// on this SAME thread via WFMO so geometry upload never races renderScene, while the
		// client can issue uploads without contending with the main cull/scene channel.
		// Null when the host is launched single-channel (e.g. the standalone --forge probes).
		HANDLE m_geomSharedMem;
		HANDLE m_geomRpcStartEvent;
		HANDLE m_geomRpcCompleteEvent;
		Parameters* m_geomParameters;
		std::vector<Vec<char>*> m_vecs;
		std::queue<VecId> m_freeVecs;
		Parameters* m_ipcParameters;

		template<typename T>
		Vec<T>& getVec(VecId id);

		bool allocVec();
		bool freeVec();
		void updateDynVis();
		bool initDistantStatics();
		bool initLandscape();
		void setWorldSpace();
		void getVisibleMeshesCoarse();
		void getVisibleMeshes();
		void getVisibleMeshesAllRanges();
		void sortVisibleSet();
		void renderInit();
		void renderFrame();
		void geomUpload();

	public:
		Server(HANDLE sharedMem, HANDLE clientProcess, HANDLE rpcStartEvent, HANDLE rpcCompleteEvent,
			HANDLE geomSharedMem = nullptr, HANDLE geomRpcStartEvent = nullptr, HANDLE geomRpcCompleteEvent = nullptr);
		~Server();
		Server(const Server&) = delete;
		Server& operator=(const Server&) = delete;

		bool init();
		bool listen();

		bool complete();
	};
}