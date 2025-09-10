#include "distantlandhlsl.h"
#include "distantland.h"
#include "configuration.h"
#include "support/log.h"
#include "hlsl_shader_manager.h"

#include <vector>
#include <string>
#include <fstream>

// Static member definitions
IDirect3DDevice9* DistantLandHLSL::device = nullptr;
bool DistantLandHLSL::enabled = false;
DistantLandHLSL::ShaderCache DistantLandHLSL::shaderCaches[SHADER_COUNT];

D3DXHANDLE DistantLandHLSL::ehWorld, DistantLandHLSL::ehView, DistantLandHLSL::ehProj;
D3DXHANDLE DistantLandHLSL::ehEyePos, DistantLandHLSL::ehSunVec, DistantLandHLSL::ehSunCol, DistantLandHLSL::ehSunAmb;
D3DXHANDLE DistantLandHLSL::ehFogColNear, DistantLandHLSL::ehFogColFar, DistantLandHLSL::ehFogStart, DistantLandHLSL::ehFogRange;
D3DXHANDLE DistantLandHLSL::ehTime, DistantLandHLSL::ehWindVec, DistantLandHLSL::ehNiceWeather;

bool DistantLandHLSL::init(IDirect3DDevice9* d) {
    device = d;
    
    // Only enable if configuration flag is set
    enabled = Configuration.UseDistantLandHLSL || true;  // Temporary: always enable for Phase 8 testing
    
    if (!enabled) {
        LOG::logline("-- Distant Land HLSL disabled via configuration");
        return true; // Success but disabled
    }
    
    if (!device) {
        LOG::logline("!! Distant Land HLSL init failed: no device");
        enabled = false;
        return false;
    }
    
    LOG::logline(">> Initializing Distant Land HLSL system");
    
    // Pre-compile common shader permutations
    ShaderPermutation basicPerm = {};
    basicPerm.useExpFog = (Configuration.MGEFlags & EXP_FOG) ? 1 : 0;
    basicPerm.useATMScatter = (Configuration.MGEFlags & USE_ATM_SCATTER) ? 1 : 0;
    basicPerm.useShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;
    
    // Pre-compile basic shaders for common cases
    getShader(SHADER_STATICS_EXTERIOR, basicPerm);
    getShader(SHADER_LAND, basicPerm);
    
    LOG::logline("<< Distant Land HLSL initialized");
    return true;
}

void DistantLandHLSL::release() {
    for (int i = 0; i < SHADER_COUNT; i++) {
        shaderCaches[i].clear();
    }
    enabled = false;
    device = nullptr;
}

char* DistantLandHLSL::loadShaderFile(const char* filename, DWORD* outFileSize) {
    std::string path = "Data Files\\shaders\\core-hlsl\\";
    path += filename;
    
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
    *outFileSize = (DWORD)size;
    return buffer;
}

void DistantLandHLSL::generateDefines(ShaderType type, const ShaderPermutation& perm, std::vector<D3DXMACRO>& defines) {
    if (perm.hasAlpha) {
        defines.push_back({"HAS_ALPHA", "1"});
    }
    if (perm.hasVCol) {
        defines.push_back({"HAS_VCOL", "1"});
    }
    if (perm.hasBones) {
        defines.push_back({"HAS_BONES", "1"});
    }
    if (perm.useATMScatter) {
        defines.push_back({"USE_ATM_SCATTER", "1"});
    }
    if (perm.useShadows) {
        defines.push_back({"USE_SHADOWS", "1"});
    }
    if (perm.useExpFog) {
        defines.push_back({"USE_EXP_FOG", "1"});
    }
    
    // Type-specific defines
    switch (type) {
        case SHADER_STATICS_EXTERIOR:
            defines.push_back({"STATICS_EXTERIOR", "1"});
            break;
        case SHADER_STATICS_INTERIOR:
            defines.push_back({"STATICS_INTERIOR", "1"});
            break;
        case SHADER_LAND:
            defines.push_back({"LANDSCAPE", "1"});
            break;
        case SHADER_GRASS:
            defines.push_back({"GRASS", "1"});
            break;
        case SHADER_SKY:
            defines.push_back({"SKY", "1"});
            break;
        case SHADER_WATER:
            defines.push_back({"WATER", "1"});
            break;
        case SHADER_DEPTH:
            defines.push_back({"DEPTH_ONLY", "1"});
            break;
    }
    
    // Null terminate
    defines.push_back({nullptr, nullptr});
}

const char* DistantLandHLSL::getShaderFilename(ShaderType type, bool isVertexShader) {
    const char* suffix = isVertexShader ? "_VS.hlsl" : "_PS.hlsl";
    
    switch (type) {
        case SHADER_STATICS_EXTERIOR:
        case SHADER_STATICS_INTERIOR:
            return isVertexShader ? "XE_Statics_VS.hlsl" : "XE_Statics_PS.hlsl";
        case SHADER_LAND:
            return isVertexShader ? "XE_Land_VS.hlsl" : "XE_Land_PS.hlsl";
        case SHADER_GRASS:
            return isVertexShader ? "XE_Grass_VS.hlsl" : "XE_Grass_PS.hlsl";
        case SHADER_SKY:
            return isVertexShader ? "XE_Sky_VS.hlsl" : "XE_Sky_PS.hlsl";
        case SHADER_WATER:
            return isVertexShader ? "XE_Water_VS.hlsl" : "XE_Water_PS.hlsl";
        case SHADER_DEPTH:
            return isVertexShader ? "XE_Depth_VS.hlsl" : "XE_Depth_PS.hlsl";
        default:
            return nullptr;
    }
}

std::unique_ptr<DistantLandHLSL::CompiledShader> DistantLandHLSL::compileShader(ShaderType type, const ShaderPermutation& perm) {
    std::vector<D3DXMACRO> d3dxDefines;
    generateDefines(type, perm, d3dxDefines);
    
    // Convert D3DXMACRO to D3D_SHADER_MACRO for D3DCompile
    std::vector<D3D_SHADER_MACRO> defines;
    for (const auto& d3dxDefine : d3dxDefines) {
        D3D_SHADER_MACRO macro = { d3dxDefine.Name, d3dxDefine.Definition };
        defines.push_back(macro);
    }
    
    auto shader = std::make_unique<CompiledShader>();
    
    // Load and compile vertex shader
    DWORD vsSize;
    char* vsSource = loadShaderFile(getShaderFilename(type, true), &vsSize);
    if (!vsSource) {
        LOG::logline("!! Failed to load vertex shader for type %d", type);
        return nullptr;
    }
    
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* vsBlobErrors = nullptr;
    
    HRESULT hr = D3DCompile(
        vsSource, vsSize, getShaderFilename(type, true), defines.data(), 
        HLSLShaderManager::getIncludeHandler(), "main", "vs_3_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &vsBlob, &vsBlobErrors
    );
    
    if (SUCCEEDED(hr)) {
        D3DXGetShaderConstantTable((DWORD*)vsBlob->GetBufferPointer(), &shader->vsConstantTable);
    }
    
    if (FAILED(hr)) {
        if (vsBlobErrors) {
            LOG::logline("!! VS compilation failed: %s", (char*)vsBlobErrors->GetBufferPointer());
            vsBlobErrors->Release();
        }
        delete[] vsSource;
        return nullptr;
    }
    
    hr = device->CreateVertexShader((DWORD*)vsBlob->GetBufferPointer(), &shader->vertexShader);
    vsBlob->Release();
    
    if (FAILED(hr)) {
        LOG::logline("!! Failed to create vertex shader");
        delete[] vsSource;
        return nullptr;
    }
    
    // Load and compile pixel shader
    DWORD psSize;
    char* psSource = loadShaderFile(getShaderFilename(type, false), &psSize);
    if (!psSource) {
        LOG::logline("!! Failed to load pixel shader for type %d", type);
        delete[] vsSource;
        return nullptr;
    }
    
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* psBlobErrors = nullptr;
    
    hr = D3DCompile(
        psSource, psSize, getShaderFilename(type, false), defines.data(), 
        HLSLShaderManager::getIncludeHandler(), "main", "ps_3_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &psBlob, &psBlobErrors
    );
    
    if (FAILED(hr)) {
        if (psBlobErrors) {
            LOG::logline("!! PS compilation failed: %s", (char*)psBlobErrors->GetBufferPointer());
            psBlobErrors->Release();
        }
        delete[] vsSource;
        delete[] psSource;
        return nullptr;
    }
    
    D3DXGetShaderConstantTable((DWORD*)psBlob->GetBufferPointer(), &shader->psConstantTable);
    delete[] vsSource;
    delete[] psSource;
    
    hr = device->CreatePixelShader((DWORD*)psBlob->GetBufferPointer(), &shader->pixelShader);
    psBlob->Release();
    
    if (FAILED(hr)) {
        LOG::logline("!! Failed to create pixel shader");
        return nullptr;
    }
    
    LOG::logline("-- Compiled HLSL shader for type %d with permutation flags: 0x%08X", type, *(DWORD*)&perm);
    return shader;
}

DistantLandHLSL::CompiledShader* DistantLandHLSL::getShader(ShaderType type, const ShaderPermutation& perm) {
    if (!enabled || type >= SHADER_COUNT) {
        return nullptr;
    }
    
    auto& cache = shaderCaches[type];
    auto it = cache.find(perm);
    
    if (it != cache.end()) {
        return it->second.get();
    }
    
    // Compile new shader
    auto compiledShader = compileShader(type, perm);
    if (!compiledShader || !compiledShader->isValid()) {
        return nullptr;
    }
    
    CompiledShader* result = compiledShader.get();
    cache[perm] = std::move(compiledShader);
    return result;
}

DistantLandHLSL::ShaderPermutation DistantLandHLSL::getCurrentPermutation() {
    ShaderPermutation perm = {};
    perm.useExpFog = (Configuration.MGEFlags & EXP_FOG) ? 1 : 0;
    perm.useATMScatter = (Configuration.MGEFlags & USE_ATM_SCATTER) ? 1 : 0;
    perm.useShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;
    return perm;
}

// Placeholder render methods - these will need to be implemented with actual rendering logic
void DistantLandHLSL::renderStaticsExterior(const ShaderPermutation& perm) {
    CompiledShader* shader = getShader(SHADER_STATICS_EXTERIOR, perm);
    if (!shader) return;
    
    device->SetVertexShader(shader->vertexShader);
    device->SetPixelShader(shader->pixelShader);
    
    // TODO: Set constants and render geometry
}

void DistantLandHLSL::renderStaticsInterior(const ShaderPermutation& perm) {
    CompiledShader* shader = getShader(SHADER_STATICS_INTERIOR, perm);
    if (!shader) return;
    
    device->SetVertexShader(shader->vertexShader);  
    device->SetPixelShader(shader->pixelShader);
    
    // TODO: Set constants and render geometry
}

void DistantLandHLSL::renderLand(const ShaderPermutation& perm) {
    CompiledShader* shader = getShader(SHADER_LAND, perm);
    if (!shader) return;
    
    device->SetVertexShader(shader->vertexShader);
    device->SetPixelShader(shader->pixelShader);
    
    // TODO: Set constants and render geometry
}

void DistantLandHLSL::renderGrass(const ShaderPermutation& perm) {
    CompiledShader* shader = getShader(SHADER_GRASS, perm);
    if (!shader) return;
    
    device->SetVertexShader(shader->vertexShader);
    device->SetPixelShader(shader->pixelShader);
    
    // TODO: Set constants and render geometry
}

void DistantLandHLSL::renderSky(const ShaderPermutation& perm) {
    CompiledShader* shader = getShader(SHADER_SKY, perm);
    if (!shader) return;
    
    device->SetVertexShader(shader->vertexShader);
    device->SetPixelShader(shader->pixelShader);
    
    // TODO: Set constants and render geometry  
}

void DistantLandHLSL::renderWater(const ShaderPermutation& perm) {
    CompiledShader* shader = getShader(SHADER_WATER, perm);
    if (!shader) return;
    
    device->SetVertexShader(shader->vertexShader);
    device->SetPixelShader(shader->pixelShader);
    
    // TODO: Set constants and render geometry
}

// Helper methods for setting constants - these would need to be connected to the actual constant tables
void DistantLandHLSL::setMatrices(const D3DXMATRIX* world, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    // TODO: Set matrix constants via constant tables
}

void DistantLandHLSL::setLighting(const D3DXVECTOR3& sunVec, const D3DXVECTOR3& sunCol, const D3DXVECTOR3& sunAmb) {
    // TODO: Set lighting constants via constant tables
}

void DistantLandHLSL::setFog(const D3DXVECTOR3& fogNear, const D3DXVECTOR3& fogFar, float fogStart, float fogRange) {
    // TODO: Set fog constants via constant tables
}

void DistantLandHLSL::setEnvironment(float time, const D3DXVECTOR2& windVec, float niceWeather) {
    // TODO: Set environment constants via constant tables
}