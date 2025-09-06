#include "shader_utils.h"
#include "support/log.h"
#include <fstream>
#include <vector>

// Static member definitions
bool ShaderUtils::dxvkDetectionCached = false;
bool ShaderUtils::dxvkDetectionResult = false;

// DXVK detection function
bool ShaderUtils::isDXVK() {
    if (dxvkDetectionCached) {
        return dxvkDetectionResult;
    }

    dxvkDetectionCached = true;
    dxvkDetectionResult = false;

    // Get d3d9.dll version info to detect DXVK
    DWORD dwHandle = 0;
    DWORD dwSize = GetFileVersionInfoSizeA("d3d9.dll", &dwHandle);
    if (dwSize == 0) {
        return false;
    }

    std::vector<BYTE> versionInfo(dwSize);
    if (!GetFileVersionInfoA("d3d9.dll", dwHandle, dwSize, versionInfo.data())) {
        return false;
    }

    // Try multiple language/codepage combinations
    const char* langCodes[] = {
        "040904b0", // US English
        "040904E4", // US English (another variant)
        "04090000", // English (neutral)
        "000004b0", // Neutral language, US English codepage
        "00000000"  // Neutral language, neutral codepage
    };

    std::string productName;
    bool foundProductName = false;

    for (const char* langCode : langCodes) {
        std::string queryPath = std::string("\\StringFileInfo\\") + langCode + "\\ProductName";
        LPVOID lpBuffer = nullptr;
        UINT uLen = 0;

        if (VerQueryValueA(versionInfo.data(), queryPath.c_str(), &lpBuffer, &uLen) && lpBuffer && uLen > 0) {
            productName = static_cast<char*>(lpBuffer);
            foundProductName = true;
            break;
        }
    }

    if (!foundProductName) {
        // Try to enumerate available language/codepage combinations
        struct LANGANDCODEPAGE {
            WORD wLanguage;
            WORD wCodePage;
        } *lpTranslate;

        UINT cbTranslate = 0;
        if (VerQueryValueA(versionInfo.data(), "\\VarFileInfo\\Translation", (LPVOID*)&lpTranslate, &cbTranslate)) {
            for (size_t i = 0; i < (cbTranslate / sizeof(LANGANDCODEPAGE)); i++) {
                char langCode[9];
                std::snprintf(langCode, sizeof(langCode), "%04x%04x", lpTranslate[i].wLanguage, lpTranslate[i].wCodePage);

                std::string queryPath = std::string("\\StringFileInfo\\") + langCode + "\\ProductName";
                LPVOID lpBuffer = nullptr;
                UINT uLen = 0;

                if (VerQueryValueA(versionInfo.data(), queryPath.c_str(), &lpBuffer, &uLen) && lpBuffer && uLen > 0) {
                    productName = static_cast<char*>(lpBuffer);
                    foundProductName = true;
                    break;
                }
            }
        }

        if (!foundProductName) {
            return false;
        }
    }

    // Check if this is DXVK
    if (productName.find("DXVK") != std::string::npos) {
        dxvkDetectionResult = true;
        LOG::logline("-- DXVK detected, fast compilation enabled");
    }

    return dxvkDetectionResult;
}

// Helper function to load shader source from file
char* ShaderUtils::loadShaderFile(const char* filename, DWORD* outFileSize) {
    std::string path;
    
    // Auto-detect path based on filename
    if (strstr(filename, "core-hlsl")) {
        path = filename;  // Already contains full path
    } else if (strstr(filename, ".hlsl")) {
        path = "Data Files\\shaders\\core-hlsl\\";
        path += filename;
    } else {
        path = "Data Files\\shaders\\core\\";
        path += filename;
    }
    
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG::logline("!! Failed to load shader file: %s", path.c_str());
        return nullptr;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    char* buffer = new char[size + 1];
    if (!file.read(buffer, size)) {
        delete[] buffer;
        LOG::logline("!! Failed to read shader file: %s", path.c_str());
        return nullptr;
    }
    
    buffer[size] = '\0';
    if (outFileSize) *outFileSize = (DWORD)size;
    return buffer;
}

void ShaderUtils::logShaderError(ID3DXBuffer* errors) {
    if (errors) {
        LOG::write("!! Shader compile errors:\n");
        LOG::write(reinterpret_cast<const char*>(errors->GetBufferPointer()));
        LOG::write("\n");
        errors->Release();
    }
    LOG::flush();
}