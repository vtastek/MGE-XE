#pragma once

#include "proxydx/d3d8header.h"
#include <unordered_map>
#include <memory>

// DistantLandHLSL - HLSL conversion system for Distant Land effects
// Phase 8: Convert Distant Land effects pipeline to HLSL with simplified permutations via ifdefs

class DistantLandHLSL {
public:
    enum ShaderType {
        SHADER_STATICS_EXTERIOR,
        SHADER_STATICS_INTERIOR, 
        SHADER_LAND,
        SHADER_GRASS,
        SHADER_SKY,
        SHADER_WATER,
        SHADER_DEPTH,
        SHADER_COUNT
    };

    struct ShaderPermutation {
        DWORD hasAlpha : 1;
        DWORD hasVCol : 1;
        DWORD hasBones : 1;
        DWORD useATMScatter : 1;
        DWORD useShadows : 1;
        DWORD useExpFog : 1;
        DWORD reserved : 26;

        bool operator==(const ShaderPermutation& other) const {
            return memcmp(this, &other, sizeof(ShaderPermutation)) == 0;
        }

        struct hasher {
            std::size_t operator()(const ShaderPermutation& k) const {
                return std::hash<uint32_t>{}(*(uint32_t*)&k);
            }
        };
    };

    struct CompiledShader {
        IDirect3DVertexShader9* vertexShader;
        IDirect3DPixelShader9* pixelShader;
        ID3DXConstantTable* vsConstantTable;
        ID3DXConstantTable* psConstantTable;
        
        CompiledShader() : vertexShader(nullptr), pixelShader(nullptr), 
                          vsConstantTable(nullptr), psConstantTable(nullptr) {}
        
        ~CompiledShader() {
            if (vertexShader) vertexShader->Release();
            if (pixelShader) pixelShader->Release();
            if (vsConstantTable) vsConstantTable->Release();
            if (psConstantTable) psConstantTable->Release();
        }
        
        bool isValid() const {
            return vertexShader && pixelShader;
        }
    };

    using ShaderCache = std::unordered_map<ShaderPermutation, std::unique_ptr<CompiledShader>, ShaderPermutation::hasher>;

private:
    static IDirect3DDevice9* device;
    static bool enabled;
    static ShaderCache shaderCaches[SHADER_COUNT];
    
    // Shader constant handles
    static D3DXHANDLE ehWorld, ehView, ehProj;
    static D3DXHANDLE ehEyePos, ehSunVec, ehSunCol, ehSunAmb;
    static D3DXHANDLE ehFogColNear, ehFogColFar, ehFogStart, ehFogRange;
    static D3DXHANDLE ehTime, ehWindVec, ehNiceWeather;
    
    static char* loadShaderFile(const char* filename, DWORD* outFileSize);
    static std::unique_ptr<CompiledShader> compileShader(ShaderType type, const ShaderPermutation& perm);
    static void generateDefines(ShaderType type, const ShaderPermutation& perm, std::vector<D3DXMACRO>& defines);
    static const char* getShaderFilename(ShaderType type, bool isVertexShader);

public:
    static bool init(IDirect3DDevice9* d);
    static void release();
    static bool isEnabled() { return enabled; }
    
    // Get or compile shader for given type and permutation
    static CompiledShader* getShader(ShaderType type, const ShaderPermutation& perm);
    
    // Render methods that mirror existing Distant Land passes
    static void renderStaticsExterior(const ShaderPermutation& perm);
    static void renderStaticsInterior(const ShaderPermutation& perm);
    static void renderLand(const ShaderPermutation& perm);
    static void renderGrass(const ShaderPermutation& perm);
    static void renderSky(const ShaderPermutation& perm);
    static void renderWater(const ShaderPermutation& perm);
    
    // Set common shader constants that mirror the effect framework
    static void setMatrices(const D3DXMATRIX* world, const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void setLighting(const D3DXVECTOR3& sunVec, const D3DXVECTOR3& sunCol, const D3DXVECTOR3& sunAmb);
    static void setFog(const D3DXVECTOR3& fogNear, const D3DXVECTOR3& fogFar, float fogStart, float fogRange);
    static void setEnvironment(float time, const D3DXVECTOR2& windVec, float niceWeather);
    
    // Helper to create permutation from current Distant Land state
    static ShaderPermutation getCurrentPermutation();
};