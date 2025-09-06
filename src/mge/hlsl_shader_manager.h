#pragma once

#include "proxydx/d3d8header.h"
#include <unordered_map>
#include <string>

// Forward declarations
class FixedFunctionShader;

// HLSL shader management functionality 
class HLSLShaderManager {
public:
    // HLSL shader compilation and caching
    static bool isDXVKDetected();
    static char* loadHLSLShaderFile(const char* filename, DWORD* outFileSize);
    
    // Shader cache management
    static void invalidateHLSLCache();
    static void checkForHLSLFileChanges();
    
    // Hot reload support
    struct CachedShaderSource {
        char* source;
        DWORD size;
        FILETIME lastWriteTime;
    };
    
    static std::unordered_map<std::string, CachedShaderSource>& getShaderSourceCache();
    static bool& getNeedsCacheReset();
    
private:
    static std::unordered_map<std::string, CachedShaderSource> shaderSourceCache;
    static bool needsCacheReset;
    
    // Make FixedFunctionShader a friend to access these utilities
    friend class FixedFunctionShader;
};