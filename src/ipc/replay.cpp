// mgeHost64.exe --forge-replay <file> [loops=3] [cpuMs=0] [window=300]   (cwd = the install dir)
//
// Plays a host recording (ipc/ipcrecord.h, MGE_IPC_RECORD) back through the same core calls the live
// server makes, with no Morrowind and no client: one D3D12 process that GPU Trace / PIX / RenderDoc can
// attach to, and the same draws and camera on every run.
//
// Pass 1 plays every record in order (init, uploads, streaming, every frame). Passes 2..loops replay
// the last `window` RenderFrames, with the geometry uploads and visibility/near-ref updates that fall
// inside that window, so a steady-state view can be measured for as long as wanted. Texture uploads and
// stream batches are pass-1 only: their slots are already resident and re-installing them would race
// the slot bookkeeping, not measure anything.
//
// cpuMs spins the replay thread before each frame, standing in for MW's frame-start work, so the
// CPU/GPU overlap question can be asked without the client.
//
// MGE_REPLAY_DUMP=<n> arms the HDR dump on replay frame n (counted across passes), for image A/Bs
// against a live dump of the same recorded frame.

#include "ipc/server.h"
#include "ipc/ipcrecord.h"
#include "ipc/dlshare.h"
#include "support/winheader.h"
#include "support/log.h"
#include "forgerender.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace IPC {
	namespace {
		const char* commandName(std::uint32_t c) {
			switch (c) {
			case Command::RenderInit: return "RenderInit";
			case Command::RenderFrame: return "RenderFrame";
			case Command::GeomUpload: return "GeomUpload";
			case Command::TexUpload: return "TexUpload";
			case Command::StreamUpload: return "StreamUpload";
			case Command::UpdateDynVis: return "UpdateDynVis";
			case Command::UpdateNearRefs: return "UpdateNearRefs";
			case Command::SetWorldSpace: return "SetWorldSpace";
			case Command::DlPrewarm: return "DlPrewarm";
			default: return "?";
			}
		}

		double qpcMs(std::int64_t ticks) {
			static const double msPerTick = [] {
				LARGE_INTEGER f; QueryPerformanceFrequency(&f);
				return 1000.0 / static_cast<double>(f.QuadPart);
			}();
			return static_cast<double>(ticks) * msPerTick;
		}

		std::int64_t qpcNow() {
			LARGE_INTEGER q; QueryPerformanceCounter(&q);
			return q.QuadPart;
		}

		void spinMs(double ms) {
			if (ms <= 0.0) {
				return;
			}
			const std::int64_t t0 = qpcNow();
			while (qpcMs(qpcNow() - t0) < ms) {
				YieldProcessor();
			}
		}

		std::uint32_t fixedU32(const Rec::Record& r, unsigned i) {
			std::uint32_t v = 0;
			if (r.fixedBytes >= (i + 1) * 4) {
				std::memcpy(&v, static_cast<const std::uint8_t*>(r.fixed) + i * 4, 4);
			}
			return v;
		}

		void streamBatch(const Rec::Record& r) {
			unsigned built = 0, failedMask = 0;
			if (!ForgeRender::streamTexturesBegin(r.blobs[0].ptr, r.blobs[0].bytes, fixedU32(r, 0), &built, &failedMask)) {
				return;   // rejected / completed at once, as live
			}
			HANDLE done = static_cast<HANDLE>(ForgeRender::streamDoneEvent());
			for (int tries = 0; tries < 600; ++tries) {
				if (done != nullptr) {
					WaitForSingleObject(done, 100);
				}
				if (ForgeRender::streamTexturesFinish(&built, &failedMask)) {
					return;
				}
			}
			LOG::logline("!! [replay] stream batch never finished");
		}
	}

	int replayMain(int argc, char** argv) {
		if (argc < 3) {
			LOG::logline("!! [replay] usage: --forge-replay <file> [loops] [cpuMs] [window]");
			return 1;
		}
		const char* path = argv[2];
		const int loops = (argc > 3) ? (std::max)(1, std::atoi(argv[3])) : 3;
		const double cpuMs = (argc > 4) ? std::atof(argv[4]) : 0.0;
		const unsigned window = (argc > 5) ? static_cast<unsigned>((std::max)(1, std::atoi(argv[5]))) : 300u;
		char dumpEnv[16] = {};
		const long dumpAt = (GetEnvironmentVariableA("MGE_REPLAY_DUMP", dumpEnv, sizeof(dumpEnv)) > 0)
			? std::strtol(dumpEnv, nullptr, 10) : -1;

		Rec::Reader rd;
		if (!rd.open(path)) {
			return 1;
		}
		if (rd.frameParamBytes != sizeof(RenderFrameParameters)) {
			LOG::logline("!! [replay] recording has RenderFrameParameters of %u bytes, this host %u: rebuilt bridge.h",
				rd.frameParamBytes, static_cast<unsigned>(sizeof(RenderFrameParameters)));
			return 1;
		}

		// Census: what the file holds, per command.
		std::vector<std::size_t> frameRecs;
		unsigned count[32] = {};
		double mb[32] = {};
		for (std::size_t i = 0; i < rd.records.size(); ++i) {
			const auto& r = rd.records[i];
			const unsigned c = (r.command < 32) ? r.command : 31;
			++count[c];
			for (std::uint32_t b = 0; b < r.blobCount; ++b) {
				mb[c] += r.blobs[b].bytes / (1024.0 * 1024.0);
			}
			if (r.command == Command::RenderFrame) {
				frameRecs.push_back(i);
			}
		}
		LOG::logline(">> [replay] %s: %zu records, %.1f MB, %zu RenderFrames; loops=%d cpuMs=%.2f window=%u",
			path, rd.records.size(), rd.size / (1024.0 * 1024.0), frameRecs.size(), loops, cpuMs, window);
		for (unsigned c = 0; c < 32; ++c) {
			if (count[c] != 0) {
				LOG::logline(">> [replay]   %-15s x%-6u %9.2f MB", commandName(c), count[c], mb[c]);
			}
		}
		if (frameRecs.empty()) {
			LOG::logline("!! [replay] no RenderFrame in the recording");
			return 1;
		}
		const std::size_t windowStart = (frameRecs.size() > window) ? frameRecs[frameRecs.size() - window] : frameRecs.front();
		LOG::flush();

		bool inited = false;
		long frameNo = 0;
		for (int pass = 0; pass < loops; ++pass) {
			const bool first = (pass == 0);
			const std::int64_t t0 = qpcNow();
			unsigned frames = 0;
			for (std::size_t i = first ? 0 : windowStart; i < rd.records.size(); ++i) {
				const auto& r = rd.records[i];
				switch (r.command) {
				case Command::RenderInit:
					if (first && !inited) {
						inited = ForgeRender::init(fixedU32(r, 0), fixedU32(r, 1), fixedU32(r, 2), fixedU32(r, 3));
						LOG::logline(">> [replay] RenderInit %ux%u %ux AF%u -> %d",
							fixedU32(r, 0), fixedU32(r, 1), fixedU32(r, 2), fixedU32(r, 3), (int)inited);
					}
					break;
				case Command::DlPrewarm:
					if (first) {
						ForgeRender::dlPrewarm();
					}
					break;
				case Command::SetWorldSpace:
					if (first) {
						char name[65] = {};
						std::memcpy(name, r.fixed, (std::min<std::uint32_t>)(r.fixedBytes, 64));
						DistantLandShare::setCurrentWorldSpace(name);
					}
					break;
				case Command::UpdateDynVis: {
					const auto* f = static_cast<const DynVisFlag*>(r.blobs[0].ptr);
					const std::uint32_t n = r.blobs[0].bytes / sizeof(DynVisFlag);
					for (std::uint32_t k = 0; k < n; ++k) {
						ForgeRender::setDistantVisGroup(f[k].groupIndex, f[k].enable);
					}
					break;
				}
				case Command::UpdateNearRefs:
					ForgeRender::setNearRefs(static_cast<const ForgeRender::NearRef*>(r.blobs[0].ptr),
						r.blobs[0].bytes / sizeof(ForgeRender::NearRef), fixedU32(r, 0));
					break;
				case Command::GeomUpload:
					if (inited && r.blobs[0].ptr != nullptr) {
						ForgeRender::uploadGeometry(r.blobs[0].ptr, r.blobs[0].bytes, fixedU32(r, 0));
					}
					break;
				case Command::TexUpload:
					if (first && inited && r.blobs[0].ptr != nullptr) {
						ForgeRender::uploadTextures(r.blobs[0].ptr, r.blobs[0].bytes, fixedU32(r, 0));
					}
					break;
				case Command::StreamUpload:
					if (first && inited) {
						streamBatch(r);
					}
					break;
				case Command::RenderFrame: {
					if (!inited) {
						break;
					}
					RenderFrameParameters params;
					std::memcpy(&params, r.fixed, sizeof(params));
					// One-shot dev edges fire where they were recorded only on pass 1's live twin; a replay
					// re-firing a shader reload or a capture every loop would measure the edge, not the frame.
					params.devReloadShaders = 0;
					params.devDistLightsToggle = 0;
					params.devGpuCapture = 0;
					params.devDumpHdr = (frameNo == dumpAt) ? 1u : 0u;
					FrameLists L = {};
					for (unsigned k = 0; k < kFrameListCount && k < r.blobCount; ++k) {
						L.ptr[k] = r.blobs[k].ptr;
						L.bytes[k] = r.blobs[k].bytes;
					}
					L.capVertBytes = (std::min)(params.capturedVertBytes, L.bytes[kListCaptured]);
					spinMs(cpuMs);
					renderFrameCore(params, L);
					++frames;
					++frameNo;
					break;
				}
				default:
					break;
				}
			}
			const double ms = qpcMs(qpcNow() - t0);
			LOG::logline(">> [replay] pass %d: %u frames in %.1f ms = %.3f ms/frame", pass + 1, frames, ms,
				frames ? ms / frames : 0.0);
			LOG::flush();
		}
		ForgeRender::shutdown();
		LOG::flush();
		return inited ? 0 : 2;
	}
}
