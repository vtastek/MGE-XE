#pragma once

#include "proxydx/d3d8header.h"

// Shader utility functions used by multiple shader subsystems
class ShaderUtils {
public:
    // DXVK detection for optimization flags
    static bool isDXVK();
    
    // Shader file loading utilities
    static char* loadShaderFile(const char* filename, DWORD* outFileSize);
    
    // Error shader creation
    static void logShaderError(ID3DXBuffer* errors);

private:
    static bool dxvkDetectionCached;
    static bool dxvkDetectionResult;
};