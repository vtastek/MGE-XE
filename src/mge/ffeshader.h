#pragma once

#include "proxydx/d3d8header.h"

#include <unordered_map>
#include <vector>



struct RenderedState {
    IDirect3DTexture9* texture;
    IDirect3DVertexBuffer9* vb;
    UINT vbOffset, vbStride;
    IDirect3DIndexBuffer9* ib;
    DWORD ibBase;
    DWORD fvf;
    DWORD zWrite, cullMode;
    DWORD vertexBlendState;
    D3DXMATRIX worldTransforms[4];
    D3DXMATRIX viewTransform;
    D3DXMATRIX worldViewTransforms[4];
    D3DCOLORVALUE diffuseMaterial;
    BYTE blendEnable, srcBlend, destBlend;
    BYTE alphaTest, alphaFunc, alphaRef;
    BYTE useLighting, useFog, matSrcDiffuse, matSrcEmissive;

    D3DPRIMITIVETYPE primType;
    UINT baseIndex, minIndex, vertCount, startIndex, primCount;
};

struct FragmentState {
    struct Stage {
        BYTE colorOp, colorArg1, colorArg2;
        BYTE alphaOp, alphaArg1, alphaArg2;
        BYTE colorArg0, alphaArg0, resultArg;
        DWORD texcoordIndex;
        DWORD texTransformFlags;
        float bumpEnvMat[2][2];
        float bumpLumiScale, bumpLumiBias;
    } stage[8];

    struct Material {
        D3DCOLORVALUE diffuse, ambient, emissive;
    } material;
};

struct LightState {
    struct Light {
        D3DLIGHTTYPE type;
        D3DCOLORVALUE diffuse;
        D3DVECTOR position;     // position / normalized direction
        D3DVECTOR viewspacePos;
        union {
            D3DVECTOR falloff;  // constant, linear, quadratic
            D3DVECTOR ambient;  // for directional lights
        };
    };

    D3DCOLORVALUE globalAmbient;
    std::unordered_map<DWORD, Light> lights;
    std::unordered_map<DWORD, bool> lightsTransformed;
    std::vector<DWORD> active;
};

class FixedFunctionShader {
    struct ShaderKey {
        DWORD uvSets : 4;
        DWORD usesSkinning : 1;
        DWORD vertexColour : 1;
        DWORD heavyLighting : 1;
        DWORD vertexMaterial : 2;
        DWORD fogMode : 2;
        DWORD activeStages : 3;
        DWORD usesBumpmap : 1;
        DWORD bumpmapStage : 3;
        DWORD usesTexgen : 1;
        DWORD projectiveTexgen : 1;
        DWORD texgenStage : 3;

        struct Stage {
            DWORD colorOp : 6;
            DWORD colorArg1 : 6;
            DWORD colorArg2 : 6;
            DWORD colorArg0 : 6;
            DWORD alphaOpMatched : 1;
            DWORD alphaOpSelect1 : 1;
            DWORD texcoordIndex : 2;
            DWORD texcoordGen : 4;
        } stage[8];

        ShaderKey() {}
        ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs);
        bool operator<(const ShaderKey& other) const;
        bool operator==(const ShaderKey& other) const;
        void log() const;

        struct hasher {
            std::size_t operator()(const ShaderKey& k) const;
        };
    };

    struct ShaderLRU {
        ID3DXEffect* effect;
        FixedFunctionShader::ShaderKey last_sk;
    };

    // Parameter caching to reduce descriptor set allocations in DXVK
    class ParameterCache {
    private:
        struct CachedParams {
            D3DXVECTOR4 materialDiffuse, materialAmbient, materialEmissive;
            IDirect3DBaseTexture9* textures[6];
            D3DXMATRIX world, worldView;
            uint32_t lightingHash;
            uint32_t frame;
        };

        static CachedParams lastParams;
        static uint32_t currentFrame;
        static bool valid;

    public:
        static void newFrame() {
            currentFrame++;
            valid = false;
        }

        static void cleanup() {
            // Release any cached texture references
            for (int i = 0; i < 6; ++i) {
                if (lastParams.textures[i]) {
                    lastParams.textures[i]->Release();
                    lastParams.textures[i] = nullptr;
                }
            }
            valid = false;
        }

        static bool needsMaterialUpdate(const FragmentState* frs) {
            return !valid || currentFrame != lastParams.frame ||
                memcmp(&lastParams.materialDiffuse, &frs->material.diffuse, sizeof(D3DXVECTOR4)) ||
                memcmp(&lastParams.materialAmbient, &frs->material.ambient, sizeof(D3DXVECTOR4)) ||
                memcmp(&lastParams.materialEmissive, &frs->material.emissive, sizeof(D3DXVECTOR4));
        }

        static bool needsTextureUpdate(const ShaderKey& sk) {
            if (!valid || currentFrame != lastParams.frame) return true;

            for (int i = 0; i < std::min((int)sk.activeStages, 6); ++i) {
                IDirect3DBaseTexture9* tex;
                device->GetTexture(i, &tex);
                if (tex != lastParams.textures[i]) {
                    if (tex) tex->Release();
                    return true;
                }
                if (tex) tex->Release();
            }
            return false;
        }

        static bool needsTransformUpdate(const RenderedState* rs) {
            return !valid || currentFrame != lastParams.frame ||
                memcmp(&lastParams.world, &rs->worldTransforms[0], sizeof(D3DXMATRIX)) ||
                memcmp(&lastParams.worldView, &rs->worldViewTransforms[0], sizeof(D3DXMATRIX));
        }

        static uint32_t hashLighting(const LightState* lightrs) {
            uint32_t hash = 0;
            for (DWORD id : lightrs->active) {
                hash ^= id + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }

        static bool needsLightingUpdate(const LightState* lightrs) {
            uint32_t newHash = hashLighting(lightrs);
            return !valid || currentFrame != lastParams.frame || newHash != lastParams.lightingHash;
        }

        static void updateCache(const RenderedState* rs, const FragmentState* frs,
            const LightState* lightrs, const ShaderKey& sk) {
            lastParams.materialDiffuse = *(D3DXVECTOR4*)&frs->material.diffuse;
            lastParams.materialAmbient = *(D3DXVECTOR4*)&frs->material.ambient;
            lastParams.materialEmissive = *(D3DXVECTOR4*)&frs->material.emissive;
            lastParams.world = rs->worldTransforms[0];
            lastParams.worldView = rs->worldViewTransforms[0];
            lastParams.lightingHash = hashLighting(lightrs);
            lastParams.frame = currentFrame;

            // Update texture cache - release old textures first
            for (int i = 0; i < 6; ++i) {
                if (lastParams.textures[i]) {
                    lastParams.textures[i]->Release();
                    lastParams.textures[i] = nullptr;
                }
            }

            // Store new texture references
            for (int i = 0; i < std::min((int)sk.activeStages, 6); ++i) {
                device->GetTexture(i, &lastParams.textures[i]);
                // Don't release here - we need to keep the reference for comparison
            }

            valid = true;
        }
    };

    static IDirect3DDevice* device;
    static ID3DXEffectPool* constantPool;
    static std::unordered_map<ShaderKey, ID3DXEffect*, ShaderKey::hasher> cacheEffects;
    static ShaderLRU shaderLRU;
    static ID3DXEffect* effectDefaultPurple;

    static D3DXHANDLE ehWorld, ehWorldView;
    static D3DXHANDLE ehVertexBlendState, ehVertexBlendPalette;
    static D3DXHANDLE ehTex0, ehTex1, ehTex2, ehTex3, ehTex4, ehTex5;
    static D3DXHANDLE ehMaterialDiffuse, ehMaterialAmbient, ehMaterialEmissive;
    static D3DXHANDLE ehLightSceneAmbient, ehLightSunDiffuse, ehLightSunDirection;
    static D3DXHANDLE ehLightDiffuse, ehLightAmbient, ehLightPosition;
    static D3DXHANDLE ehLightFalloffQuadratic, ehLightFalloffLinear, ehLightFalloffConstant;
    static D3DXHANDLE ehTexgenTransform, ehBumpMatrix, ehBumpLumiScaleBias;

    static float sunMultiplier, ambMultiplier;

    static ID3DXEffect* generateMWShader(const ShaderKey& sk);

public:
    static bool init(IDirect3DDevice* d, ID3DXEffectPool* pool);
    static void precache();
    static void updateLighting(float sunMult, float ambMult);
    static void renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs);
    static void release();
    static void newFrame() { ParameterCache::newFrame(); }  // Public accessor for parameter cache
    static void cleanupParameterCache() { ParameterCache::cleanup(); }  // Public cleanup accessor
};