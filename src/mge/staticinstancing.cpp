// Distant-static instancing implementations. Kept here so the rest of
// the distant-content pipeline (distantland.{h,cpp}, renderexterior.cpp,
// renderdepth.cpp) doesn't carry instancing-specific code.

#include "staticinstancing.h"

#include "configuration.h"
#include "distantland.h"
#include "distantshader.h"
#include "msocclient.h"
#include "mwbridge.h"
#include "phasetimers.h"
#include "support/log.h"

#include <utility>
#include <vector>

namespace StaticInstancing {

// Per-frame instance vertex buffer. Holds packed 4x3 transposed world
// matrices, one per visible instance, organized into batched groups
// that share (vBuffer, tex, hasAlpha, animateUV). Created once at init
// time, refilled with D3DLOCK_DISCARD each frame.
constexpr int StaticInstStride    = 48;
constexpr int MaxStaticInstances  = 32768;

static IDirect3DVertexBuffer9* vbStaticInstances = nullptr;

// Per-frame draw-call list — group head + instance count. Filled by
// buildVB, consumed by renderColor / renderDepth.
static std::vector<std::pair<RenderMesh, int>> batchedStatics;

bool initVB(IDirect3DDevice9* device) {
    if (vbStaticInstances) {
        return true;   // already initialized
    }
    HRESULT hr = device->CreateVertexBuffer(
        MaxStaticInstances * StaticInstStride,
        D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
        0,
        D3DPOOL_DEFAULT,
        &vbStaticInstances,
        nullptr);
    if (hr != D3D_OK) {
        LOG::logline("!! StaticInstancing::initVB failed (hr=0x%08x)", hr);
        vbStaticInstances = nullptr;
        return false;
    }
    return true;
}

void shutdownVB() {
    if (vbStaticInstances) {
        vbStaticInstances->Release();
        vbStaticInstances = nullptr;
    }
    batchedStatics.clear();
    batchedStatics.shrink_to_fit();
}

// Walk the sorted visible set, group consecutive meshes sharing
// (vBuffer, tex, hasAlpha, animateUV), and pack each group's transforms
// as transposed 4x3 matrices into the per-frame instance VB.
// batchedStatics is populated with (groupHead, instanceCount) pairs.
//
// Mirrors buildGrassInstanceVB in rendergrass.cpp; difference: grass
// shares pipeline state across all of its instances so it groups only
// on vBuffer, statics need a wider key because each can have its own
// texture / alpha / UV-anim state.
//
// Consumes DistantLand::msocOccluded — populated by
// applyMSOCToDistantStatics — to skip MSOC-culled instances.
template<class T>
void buildVB(VisibleSet<T>& staticSet) {
    MGE_SCOPED_TIMER("buildStaticInstanceVB");
    batchedStatics.clear();

    if (staticSet.Empty()) {
        return;
    }

    if (staticSet.Size() > MaxStaticInstances) {
        static bool warnOnce = true;
        if (warnOnce) {
            LOG::logline("!! Too many distant-static instances. (%d elements, limit %d)", staticSet.Size(), MaxStaticInstances);
            LOG::logline("!! Some distant statics will not render. Reduce draw distance or lower static density.");
            warnOnce = false;
        }
        staticSet.Truncate(MaxStaticInstances);
    }

    // Snapshot the group head BY VALUE. We can't hold a pointer into the
    // source visible-set: with shared-memory IPC, visDistantShared iterates
    // through a remapping window (see IpcClientVector::next()), so any
    // reference into the source is invalidated on the next advance.
    RenderMesh groupHead{};
    bool haveGroupHead = false;
    float* vbwrite = nullptr;
    int instanceCount = 0;

    HRESULT hr = vbStaticInstances->Lock(0, StaticInstStride * staticSet.Size(), (void**)&vbwrite, D3DLOCK_DISCARD);
    if (hr != D3D_OK || vbwrite == 0) {
        return;
    }

    const bool cullByOcclusion =
        Configuration.UseOcclusionCulling
        && MSOCClient::isAvailable()
        && MSOCClient::isMaskReady();

    staticSet.Reset();
    unsigned idx = 0;
    const bool haveMSOCMask = !DistantLand::msocOccluded.empty();
    while (!staticSet.AtEnd()) {
        const auto& m = staticSet.Next();

        // Consult the prebuilt MSOC verdict mask — populated by
        // applyMSOCToDistantStatics in cullDistantStatics. Mask order
        // matches the visible-set iteration order, so idx is the key.
        if (haveMSOCMask && DistantLand::msocOccluded[idx]) {
            ++idx;
            continue;
        }

        if (!haveGroupHead) {
            groupHead = m;
            haveGroupHead = true;
        } else if (groupHead.vBuffer != m.vBuffer
                   || groupHead.tex != m.tex
                   || groupHead.hasAlpha != m.hasAlpha
                   || groupHead.animateUV != m.animateUV) {
            batchedStatics.push_back(std::make_pair(groupHead, instanceCount));
            groupHead = m;
            instanceCount = 0;
        }

        // Pack per-instance transform as a transposed 4x3 matrix (12 floats).
        const D3DMATRIX* world = &m.transform;
        vbwrite[0]  = world->_11; vbwrite[1]  = world->_21; vbwrite[2]  = world->_31; vbwrite[3]  = world->_41;
        vbwrite[4]  = world->_12; vbwrite[5]  = world->_22; vbwrite[6]  = world->_32; vbwrite[7]  = world->_42;
        vbwrite[8]  = world->_13; vbwrite[9]  = world->_23; vbwrite[10] = world->_33; vbwrite[11] = world->_43;
        vbwrite += 12;
        instanceCount++;
        ++idx;
    }
    if (haveGroupHead) {
        batchedStatics.push_back(std::make_pair(groupHead, instanceCount));
    }

    vbStaticInstances->Unlock();

    // Batch-size diagnostic, gated by LogDistantPipeline. The cull-rate
    // stats (-- MSOC cull:) live in applyMSOCToDistantStatics; the
    // batched-instance distribution is specific to the instancing path
    // so it lives here.
    if (Configuration.LogDistantPipeline && cullByOcclusion) {
        static int diagFrameCounter = 0;
        if ((diagFrameCounter++ % 60) == 0) {
            int batchMin = 0, batchMax = 0, batchSum = 0;
            if (!batchedStatics.empty()) {
                batchMin = batchedStatics[0].second;
                batchMax = batchedStatics[0].second;
                for (const auto& b : batchedStatics) {
                    if (b.second < batchMin) batchMin = b.second;
                    if (b.second > batchMax) batchMax = b.second;
                    batchSum += b.second;
                }
            }
            const int batchAvg = !batchedStatics.empty()
                ? batchSum / (int)batchedStatics.size() : 0;
            LOG::logline(
                "-- DL batches: count=%u  instances=%d  min=%d  avg=%d  max=%d",
                (unsigned)batchedStatics.size(),
                batchSum, batchMin, batchAvg, batchMax);
        }
    }
}

template void buildVB(VisibleSet<StlVector>& staticSet);
template void buildVB(VisibleSet<IpcClientVector>& staticSet);

// Common core of the two instanced-statics entry points.
// Iterates batchedStatics, rebinds per-group state (texture, alpha test,
// animate-uv), and issues one instanced DrawIndexedPrimitive per group.
//
// Effect-variable writes go through `effect` (the main effect that owns
// the shared effect-pool parameters ehTex0 / ehHasVCol). CommitChanges
// is called on `e` so the currently-running pass (which may live in
// `effect` OR `effectDepth`) picks them up. Mirrors the pattern used in
// renderGrassCommon.
static void renderCommon(ID3DXEffect* e) {
    MGE_SCOPED_TIMER("renderDistantStaticsInstancedCommon");
    DistantLand::device->SetVertexDeclaration(DistantLand::GrassDecl);

    // Reset animate_uv so the first group picks up a fresh state.
    DistantLand::effect->SetBool(DistantLand::ehHasVCol, false);
    e->CommitChanges();

    IDirect3DTexture9* lastTexture = nullptr;
    bool lastAnimateUV = false;
    bool lastHasAlpha = false;
    int firstInstance = 0;

    for (const auto& batch : batchedStatics) {
        const RenderMesh& mesh = batch.first;
        int count = batch.second;

        // Per-group state setup (rebind-filtered by the sort + group boundary rules).
        if (lastTexture != mesh.tex) {
            DistantLand::effect->SetTexture(DistantLand::ehTex0, mesh.tex);
            DistantLand::device->SetRenderState(D3DRS_ALPHATESTENABLE, mesh.hasAlpha);
            lastTexture = mesh.tex;
            lastHasAlpha = mesh.hasAlpha;
        } else if (lastHasAlpha != mesh.hasAlpha) {
            DistantLand::device->SetRenderState(D3DRS_ALPHATESTENABLE, mesh.hasAlpha);
            lastHasAlpha = mesh.hasAlpha;
        }

        if (lastAnimateUV != mesh.animateUV) {
            DistantLand::effect->SetBool(DistantLand::ehHasVCol, mesh.animateUV);
            lastAnimateUV = mesh.animateUV;
        }

        e->CommitChanges();

        // Bind per-group static VB + shared instance VB and issue one instanced draw.
        DistantLand::device->SetIndices(mesh.iBuffer);
        DistantLand::device->SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | count);
        DistantLand::device->SetStreamSource(0, mesh.vBuffer, 0, SIZEOFSTATICVERT);
        DistantLand::device->SetStreamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1);
        DistantLand::device->SetStreamSource(1, vbStaticInstances, StaticInstStride * firstInstance, StaticInstStride);
        DistantLand::device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, mesh.verts, 0, mesh.faces);

        firstInstance += count;
    }

    // Restore non-instanced state so later unrelated draws don't inherit the instance freq.
    DistantLand::device->SetStreamSourceFreq(0, 1);
    DistantLand::device->SetStreamSourceFreq(1, 1);
    DistantLand::device->SetStreamSource(1, NULL, 0, 0);
}

void renderColor() {
    MGE_SCOPED_TIMER("renderDistantStaticsInstanced");
    if (batchedStatics.empty()) {
        return;
    }

    if (!MWBridge::get()->IsExterior()) {
        // Same architectural clip-plane workaround as renderDistantStatics.
        float clipAt = DistantLand::nearViewRange - 768.0f;
        D3DXPLANE clipPlane(0, 0, clipAt,
                            -(DistantLand::mwProj._33 * clipAt + DistantLand::mwProj._43));
        DistantLand::device->SetClipPlane(0, clipPlane);
        DistantLand::device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);
    }

    renderCommon(DistantLand::effect);

    DistantLand::device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
}

void renderDepth() {
    MGE_SCOPED_TIMER("renderDistantStaticsInstancedZ");
    if (batchedStatics.empty()) {
        return;
    }

    renderCommon(DistantLand::effectDepth);
}

} // namespace StaticInstancing
