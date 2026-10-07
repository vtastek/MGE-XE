#pragma once

// Host-side IPC recording (tasks/forge-ipc-replay.md in the devkit).
//
// The host records the ARGUMENTS of every core call an RPC makes (renderScene's resolved lists,
// uploadGeometry's blob, …), not the client's shared memory. Replay (`mgeHost64.exe --forge-replay`)
// feeds them straight back into the same calls, so the host runs the recorded frames with no
// Morrowind, no client, and no cross-process seam: a single D3D12 process that GPU Trace, PIX and
// RenderDoc can all attach to.
//
// File: "MGEREC01" + u32 sizeof(RenderFrameParameters), then records:
//   u32 magic 'MREC', u32 command, u64 qpc, u32 fixedBytes, u32 blobCount, fixed bytes,
//   then per blob: u32 bytes, zero padding to a 64-byte file offset, the bytes.
// A truncated tail (host killed mid-write) ends the replay at the last whole record.

#include <cstdint>
#include <vector>

namespace IPC {
	namespace Rec {
		static constexpr std::uint32_t kMagic = 0x4345524Du;   // 'MREC'
		static constexpr std::uint32_t kMaxBlobs = 12;

		struct Blob { const void* ptr; std::uint32_t bytes; };

		// Recording (env MGE_IPC_RECORD=<path>, MGE_IPC_RECORD_FRAMES=<n>, default 1200 RenderFrames).
		void openFromEnv();
		bool active();
		void write(std::uint32_t command, const void* fixed, std::uint32_t fixedBytes,
			const Blob* blobs, std::uint32_t blobCount);
		// After a RenderFrame record: counts it, flushes periodically, closes at the frame budget.
		void frameDone();

		// Reading. The file stays mapped for the reader's lifetime, so blob pointers are stable.
		struct Record {
			std::uint32_t command;
			std::uint64_t qpc;
			const void* fixed;
			std::uint32_t fixedBytes;
			std::uint32_t blobCount;
			Blob blobs[kMaxBlobs];
		};
		struct Reader {
			void* file = nullptr;
			void* mapping = nullptr;
			const std::uint8_t* base = nullptr;
			std::uint64_t size = 0;
			std::uint32_t frameParamBytes = 0;
			std::vector<Record> records;
			bool open(const char* path);
			~Reader();
		};
	}
}
