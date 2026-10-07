#include "ipc/ipcrecord.h"
#include "ipc/bridge.h"
#include "support/winheader.h"
#include "support/log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace IPC {
	namespace Rec {
		static std::FILE* g_file = nullptr;
		static std::uint64_t g_offset = 0;
		static std::uint32_t g_frames = 0;
		static std::uint32_t g_maxFrames = 1200;
		static std::uint64_t g_bytesAtOpen = 0;

		static const char kHeader[8] = { 'M', 'G', 'E', 'R', 'E', 'C', '0', '1' };

		static void put(const void* p, std::uint64_t n) {
			if (n != 0) {
				std::fwrite(p, 1, static_cast<size_t>(n), g_file);
				g_offset += n;
			}
		}

		static void padTo64() {
			static const std::uint8_t zeros[64] = {};
			const std::uint64_t pad = (64 - (g_offset & 63)) & 63;
			put(zeros, pad);
		}

		void openFromEnv() {
			char path[MAX_PATH] = {};
			if (GetEnvironmentVariableA("MGE_IPC_RECORD", path, sizeof(path)) == 0 || g_file != nullptr) {
				return;
			}
			if (char n[16] = {}; GetEnvironmentVariableA("MGE_IPC_RECORD_FRAMES", n, sizeof(n)) > 0) {
				const long v = std::strtol(n, nullptr, 10);
				if (v > 0) {
					g_maxFrames = static_cast<std::uint32_t>(v);
				}
			}
			if (fopen_s(&g_file, path, "wb") != 0 || g_file == nullptr) {
				g_file = nullptr;
				LOG::logline("!! [rec] cannot open %s for writing", path);
				return;
			}
			// Large stdio buffer: a cell-load burst of GeomUploads must not turn into many small writes
			// on the RPC thread the client is blocked on.
			std::setvbuf(g_file, nullptr, _IOFBF, 16u << 20);
			put(kHeader, sizeof(kHeader));
			const std::uint32_t frameBytes = sizeof(RenderFrameParameters);
			put(&frameBytes, sizeof(frameBytes));
			LOG::logline(">> [rec] recording host RPCs to %s for %u RenderFrames", path, g_maxFrames);
		}

		bool active() {
			return g_file != nullptr;
		}

		void write(std::uint32_t command, const void* fixed, std::uint32_t fixedBytes,
			const Blob* blobs, std::uint32_t blobCount) {
			if (g_file == nullptr) {
				return;
			}
			LARGE_INTEGER q; QueryPerformanceCounter(&q);
			const std::uint32_t magic = kMagic;
			const std::uint64_t qpc = static_cast<std::uint64_t>(q.QuadPart);
			put(&magic, 4);
			put(&command, 4);
			put(&qpc, 8);
			put(&fixedBytes, 4);
			put(&blobCount, 4);
			put(fixed, fixedBytes);
			for (std::uint32_t i = 0; i < blobCount; ++i) {
				const std::uint32_t n = (blobs[i].ptr != nullptr) ? blobs[i].bytes : 0;
				put(&n, 4);
				padTo64();
				put(blobs[i].ptr, n);
			}
		}

		void frameDone() {
			if (g_file == nullptr) {
				return;
			}
			++g_frames;
			// Periodic flush: the harness kills the host, and an unflushed 16 MB tail is most of a
			// steady-state window.
			if ((g_frames % 30) == 0) {
				std::fflush(g_file);
			}
			if (g_frames >= g_maxFrames) {
				std::fclose(g_file);
				g_file = nullptr;
				LOG::logline(">> [rec] recording closed: %u RenderFrames, %.1f MB",
					g_frames, static_cast<double>(g_offset) / (1024.0 * 1024.0));
				LOG::flush();
			}
		}

		bool Reader::open(const char* path) {
			HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
				FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
			if (f == INVALID_HANDLE_VALUE) {
				LOG::logline("!! [replay] cannot open %s", path);
				return false;
			}
			file = f;
			LARGE_INTEGER sz = {};
			GetFileSizeEx(f, &sz);
			size = static_cast<std::uint64_t>(sz.QuadPart);
			if (size < sizeof(kHeader) + 4) {
				LOG::logline("!! [replay] %s is too short to be a recording", path);
				return false;
			}
			mapping = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
			if (mapping == nullptr) {
				LOG::winerror("[replay] CreateFileMapping");
				return false;
			}
			base = static_cast<const std::uint8_t*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
			if (base == nullptr) {
				LOG::winerror("[replay] MapViewOfFile");
				return false;
			}
			if (std::memcmp(base, kHeader, sizeof(kHeader)) != 0) {
				LOG::logline("!! [replay] %s has no MGEREC01 header", path);
				return false;
			}
			std::memcpy(&frameParamBytes, base + sizeof(kHeader), 4);

			std::uint64_t off = sizeof(kHeader) + 4;
			auto have = [&](std::uint64_t n) { return off + n <= size; };
			auto u32 = [&]() { std::uint32_t v; std::memcpy(&v, base + off, 4); off += 4; return v; };
			while (have(24)) {
				Record r = {};
				if (u32() != kMagic) {
					LOG::logline("!! [replay] bad record magic at offset %llu; stopping", off - 4);
					break;
				}
				r.command = u32();
				std::memcpy(&r.qpc, base + off, 8); off += 8;
				r.fixedBytes = u32();
				r.blobCount = u32();
				if (r.blobCount > kMaxBlobs || !have(r.fixedBytes)) {
					break;
				}
				r.fixed = base + off;
				off += r.fixedBytes;
				bool whole = true;
				for (std::uint32_t i = 0; i < r.blobCount; ++i) {
					if (!have(4)) { whole = false; break; }
					const std::uint32_t n = u32();
					off += (64 - (off & 63)) & 63;
					if (!have(n)) { whole = false; break; }
					r.blobs[i].ptr = (n != 0) ? base + off : nullptr;
					r.blobs[i].bytes = n;
					off += n;
				}
				if (!whole) {
					break;   // truncated tail
				}
				records.push_back(r);
			}
			return true;
		}

		Reader::~Reader() {
			if (base != nullptr) {
				UnmapViewOfFile(base);
			}
			if (mapping != nullptr) {
				CloseHandle(mapping);
			}
			if (file != nullptr) {
				CloseHandle(file);
			}
		}
	}
}
