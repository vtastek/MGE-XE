// Streaming TES3 plugin reader for the distant-land baker.
//
// The whole point of this file is that NOTHING here accumulates per reference. A plugin is mapped,
// walked, and released; callers see one record and one subrecord at a time. That is what keeps the
// baker's peak memory a function of the WORLD size (cells) rather than the CONTENT size
// (placements) -- see tasks/dl-gen-ownership.md section 1 for the 2.3 GB that the old design needed.
#pragma once

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <string>

namespace bake {

// A read-only memory-mapped plugin. 264 MB plugins exist (lush3_TR_merged.esp), so the file is
// mapped rather than read: no copy, and the pages the walk never touches are never faulted in.
struct EspFile {
    HANDLE         hFile = INVALID_HANDLE_VALUE;
    HANDLE         hMap  = nullptr;
    const uint8_t* data  = nullptr;
    uint64_t       size  = 0;

    bool open(const char* path) {
        close();
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) { return false; }
        LARGE_INTEGER li{};
        if (!GetFileSizeEx(hFile, &li) || li.QuadPart == 0) { close(); return false; }
        size = (uint64_t)li.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { close(); return false; }
        data = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data) { close(); return false; }
        return true;
    }

    void close() {
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = nullptr; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
        size = 0;
    }

    ~EspFile() { close(); }
};

inline uint32_t rdU32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline int32_t  rdI32(const uint8_t* p) { int32_t  v; std::memcpy(&v, p, 4); return v; }
inline float    rdF32(const uint8_t* p) { float    v; std::memcpy(&v, p, 4); return v; }

inline bool tagIs(const uint8_t* p, const char* t) {
    return p[0] == (uint8_t)t[0] && p[1] == (uint8_t)t[1] && p[2] == (uint8_t)t[2] && p[3] == (uint8_t)t[3];
}

// A TES3 record: 16-byte header (tag, size, unused, flags) then `size` bytes of subrecords.
struct RecordIter {
    const uint8_t* p   = nullptr;
    const uint8_t* end = nullptr;

    const uint8_t* tag  = nullptr;   // 4 bytes, not NUL-terminated
    const uint8_t* body = nullptr;
    uint32_t       size = 0;

    explicit RecordIter(const EspFile& f) : p(f.data), end(f.data + f.size) {}

    bool next() {
        if (p + 16 > end) { return false; }
        tag  = p;
        size = rdU32(p + 4);
        body = p + 16;
        if (body + size > end) { return false; }     // truncated tail: stop cleanly
        p = body + size;
        return true;
    }
};

// A subrecord inside one record: 8-byte header (tag, size) then `size` bytes.
struct SubIter {
    const uint8_t* p   = nullptr;
    const uint8_t* end = nullptr;

    const uint8_t* tag  = nullptr;
    const uint8_t* body = nullptr;
    uint32_t       size = 0;

    SubIter(const uint8_t* b, uint32_t n) : p(b), end(b + n) {}

    bool next() {
        if (p + 8 > end) { return false; }
        tag  = p;
        size = rdU32(p + 4);
        body = p + 8;
        if (body + size > end) { return false; }
        p = body + size;
        return true;
    }
};

// Plugin strings are NUL-padded cp1252. Object ids and model paths are compared case-insensitively
// everywhere in the DL pipeline, so they are lowercased once here rather than at every lookup.
inline std::string subString(const uint8_t* body, uint32_t size) {
    uint32_t n = 0;
    while (n < size && body[n] != 0) { ++n; }
    std::string s((const char*)body, n);
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') { c = (char)(c - 'A' + 'a'); }
    }
    return s;
}

}  // namespace bake
