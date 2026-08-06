// mgeHost64 — spatiotemporal blue-noise mask, embedded (see stbn_mask.cpp, which is GENERATED).
//
// A 64 x (64*16) R8G8 atlas: 16 time slices of a 64x64 mask, stacked vertically. Both channels are
// blue over SPACE within a slice and blue over TIME along a pixel's 16-frame column — the two
// properties a plain blue-noise texture and a plain 3D blue noise respectively do not have.
//
// Why it is embedded rather than a file: it is 128 KB of incompressible data that must match the
// shader's index arithmetic exactly, and shipping it as an asset means one more thing that can go
// missing from an install and fail at runtime instead of at link time.
#pragma once

extern const unsigned int  kStbnWidth;
extern const unsigned int  kStbnHeight;
extern const unsigned int  kStbnSlices;
extern const unsigned int  kStbnBytes;
extern const unsigned char kStbnMaskRG8[];
