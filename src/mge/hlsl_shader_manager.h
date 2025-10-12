#pragma once

#include "proxydx/d3d8header.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <d3dcompiler.h>

// Forward declarations
class FixedFunctionShader;

// Custom include handler for HLSL shader #include support
class HLSLIncludeHandler : public ID3DInclude {
public:
    HRESULT __stdcall Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) override;
    HRESULT __stdcall Close(LPCVOID pData) override;

private:
    std::mutex includeMutex; // Protect loadedIncludes from concurrent access
    std::unordered_map<LPCVOID, char*> loadedIncludes; // Track allocated memory for cleanup
};

// HLSL shader management functionality 
class HLSLShaderManager {
public:
    // HLSL shader compilation and caching
    static bool isDXVKDetected();
    static char* loadHLSLShaderFile(const char* filename, DWORD* outFileSize);
    
    // Include handler for #include support
    static HLSLIncludeHandler* getIncludeHandler();
    
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