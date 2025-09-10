// HLSL shader management functionality - implementation
#include "hlsl_shader_manager.h"
#include "shader_utils.h"
#include "support/log.h"
#include <algorithm>

// Static member definitions
std::unordered_map<std::string, HLSLShaderManager::CachedShaderSource> HLSLShaderManager::shaderSourceCache;
bool HLSLShaderManager::needsCacheReset = false;
static HLSLIncludeHandler includeHandler;

bool HLSLShaderManager::isDXVKDetected() {
    return ShaderUtils::isDXVK();
}

char* HLSLShaderManager::loadHLSLShaderFile(const char* filename, DWORD* outFileSize) {
    std::string key(filename);
    
    // Check if we have cached version
    auto it = shaderSourceCache.find(key);
    if (it != shaderSourceCache.end()) {
        // Check if file has been modified since we cached it
        WIN32_FIND_DATAA findData;
        HANDLE hFind = FindFirstFileA(filename, &findData);
        if (hFind != INVALID_HANDLE_VALUE) {
            FindClose(hFind);
            if (CompareFileTime(&it->second.lastWriteTime, &findData.ftLastWriteTime) == 0) {
                // File unchanged, return cached version
                if (outFileSize) *outFileSize = it->second.size;
                char* cachedCopy = new char[it->second.size + 1];
                memcpy(cachedCopy, it->second.source, it->second.size);
                cachedCopy[it->second.size] = '\0';
                return cachedCopy;
            } else {
                // File changed! Mark for recompilation but don't clear cache immediately
                delete[] it->second.source;
                shaderSourceCache.erase(it);
                // Set flag to clear cache after current compilation operations complete
                needsCacheReset = true;
            }
        }
    }
    
    // Get file attributes for initial caching
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(filename, &findData);
    if (hFind == INVALID_HANDLE_VALUE) {
        LOG::logline("!! HLSL file not found: %s", filename);
        return nullptr;
    }
    FindClose(hFind);
    
    // Read file from disk
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        LOG::logline("!! HLSL file read error: %s", filename);
        return nullptr;
    }
    
    DWORD fileSize = GetFileSize(hFile, nullptr);
    char* shaderSource = new char[fileSize + 1];
    DWORD bytesRead;
    ReadFile(hFile, shaderSource, fileSize, &bytesRead, nullptr);
    shaderSource[fileSize] = '\0';
    CloseHandle(hFile);
    
    // Cache the result
    CachedShaderSource cached;
    cached.source = new char[fileSize + 1];
    memcpy(cached.source, shaderSource, fileSize);
    cached.source[fileSize] = '\0';
    cached.size = fileSize;
    cached.lastWriteTime = findData.ftLastWriteTime;
    shaderSourceCache[key] = cached;
    
    if (outFileSize) *outFileSize = fileSize;
    return shaderSource;
}

void HLSLShaderManager::invalidateHLSLCache() {
    // Clear shader source cache for hot reloading
    for (auto& i : shaderSourceCache) {
        delete[] i.second.source;
    }
    shaderSourceCache.clear();
    needsCacheReset = true;
    
    LOG::logline("-- HLSL shader source cache invalidated for hot reload");
}

void HLSLShaderManager::checkForHLSLFileChanges() {
    // Check if any cached shader files have been modified
    const char* shaderFiles[] = {
        "Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_VS.hlsl",
        "Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_PS.hlsl"
    };
    
    bool anyChanged = false;
    for (const char* filename : shaderFiles) {
        std::string key(filename);
        auto it = shaderSourceCache.find(key);
        if (it != shaderSourceCache.end()) {
            // Check file modification time
            WIN32_FIND_DATAA findData;
            HANDLE hFind = FindFirstFileA(filename, &findData);
            if (hFind != INVALID_HANDLE_VALUE) {
                FindClose(hFind);
                if (CompareFileTime(&it->second.lastWriteTime, &findData.ftLastWriteTime) != 0) {
                    anyChanged = true;
                    break;
                }
            }
        }
    }
    
    if (anyChanged) {
        LOG::logline("-- HLSL shader files changed, invalidating cache for hot reload");
        invalidateHLSLCache();
    }
}

std::unordered_map<std::string, HLSLShaderManager::CachedShaderSource>& HLSLShaderManager::getShaderSourceCache() {
    return shaderSourceCache;
}

bool& HLSLShaderManager::getNeedsCacheReset() {
    return needsCacheReset;
}

HLSLIncludeHandler* HLSLShaderManager::getIncludeHandler() {
    return &includeHandler;
}

// HLSLIncludeHandler implementation
HRESULT HLSLIncludeHandler::Open(D3D_INCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID* ppData, UINT* pBytes) {
    // Build full path for include file
    std::string includePath;
    
    if (IncludeType == D3D_INCLUDE_LOCAL) {
        // Local includes (#include "file.hlsl") - look in shaders directory
        includePath = "Data Files\\shaders\\core-hlsl\\";
        includePath += pFileName;
    } else if (IncludeType == D3D_INCLUDE_SYSTEM) {
        // System includes (#include <file.hlsl>) - also look in shaders directory for now
        includePath = "Data Files\\shaders\\core-hlsl\\";
        includePath += pFileName;
    } else {
        LOG::logline("!! HLSL Include: Unknown include type for %s", pFileName);
        return E_FAIL;
    }
    
    // Load the include file
    DWORD fileSize = 0;
    char* fileData = HLSLShaderManager::loadHLSLShaderFile(includePath.c_str(), &fileSize);
    
    if (!fileData) {
        LOG::logline("!! HLSL Include: Failed to load %s", includePath.c_str());
        return E_FAIL;
    }
    
    // Store for cleanup
    loadedIncludes[fileData] = fileData;
    
    // Return the data
    *ppData = fileData;
    *pBytes = fileSize;
    
    return S_OK;
}

HRESULT HLSLIncludeHandler::Close(LPCVOID pData) {
    // Find and free the allocated memory
    auto it = loadedIncludes.find(pData);
    if (it != loadedIncludes.end()) {
        delete[] it->second;
        loadedIncludes.erase(it);
        return S_OK;
    }
    
    LOG::logline("!! HLSL Include: Failed to find include data for cleanup");
    return E_FAIL;
}