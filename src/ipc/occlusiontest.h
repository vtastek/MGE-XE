#pragma once

// HOST-ONLY (mgeHost64). Reconstructs the plugin's snapshot MOC mask from a
// shipped blob (see occlusionmask.h) and runs the EXACT sphere→NDC→TestRect
// query the plugin runs in-process, so the host's distant-statics occlusion
// verdicts are bit-identical to MGE's current in-process cull. Requires the
// vendored Intel MOC (3rdparty/msoc) on the include path. Not included by the
// 32-bit d3d8 build (which only needs the layout header).

#include "occlusionmask.h"
#include "MaskedOcclusionCulling.h"

namespace OcclusionMask {

    // Owns a MaskedOcclusionCulling instance reconstructed from a blob.
    class HostMask {
    public:
        ~HostMask() {
            if (moc_) { MaskedOcclusionCulling::Destroy(moc_); moc_ = nullptr; }
        }

        // Copy a shipped blob into the instance. Returns false (and clears
        // ready) on malformed / version-mismatch / stale-snapshot / layout-
        // mismatch — every failure path leaves the host culling NOTHING.
        bool load(const void* blob, int blobBytes) {
            ready_ = false;
            if (!blob || blobBytes < (int)sizeof(Header)) return false;
            const Header* h = static_cast<const Header*>(blob);
            if (h->version != kVersion) return false;
            if (h->headerBytes < sizeof(Header)) return false;
            if ((long long)h->headerBytes + h->zbufBytes > blobBytes) return false;
            if (!h->ready) return false;          // stale → keep everything

            // (Re)create to match the producer's SIMD tier + resolution. Same
            // physical CPU ⇒ the tier is available ⇒ identical ZTile layout.
            if (moc_ == nullptr || implCreated_ != h->impl ||
                maskW_ != h->maskW || maskH_ != h->maskH) {
                if (moc_) { MaskedOcclusionCulling::Destroy(moc_); moc_ = nullptr; }
                moc_ = MaskedOcclusionCulling::Create(
                    (MaskedOcclusionCulling::Implementation)h->impl);
                if (!moc_) return false;
                moc_->SetResolution((unsigned)h->maskW, (unsigned)h->maskH);
                moc_->SetNearClipPlane(h->nearClipW);
                implCreated_ = h->impl; maskW_ = h->maskW; maskH_ = h->maskH;
            }
            // Layout guard: a mismatch means the raw bytes can't be trusted.
            if ((std::uint32_t)moc_->GetHiZBufferSize() != h->zbufBytes) return false;
            moc_->SetHiZBuffer(zbuffer(h));

            hdr_ = *h;
            ready_ = true;
            return true;
        }

        bool ready() const { return ready_; }
        const Header& header() const { return hdr_; }

        // Faithful port of the plugin's testSphereVisible. Returns the MOC
        // CullingResult: VISIBLE(0) keep, OCCLUDED(1) cull, VIEW_CULLED(3) keep
        // (sphere outside the mask frustum — the host's distant frustum is wider).
        MaskedOcclusionCulling::CullingResult testSphere(
            float cx, float cy, float cz, float radius) const {
            if (!ready_ || !moc_) return MaskedOcclusionCulling::VISIBLE;
            if (hdr_.minRadius > 0.0f && radius < hdr_.minRadius)
                return MaskedOcclusionCulling::VISIBLE;

            const float* m = hdr_.viewProj;
            const float clipX = cx*m[0] + cy*m[4] + cz*m[8]  + m[12];
            const float clipY = cx*m[1] + cy*m[5] + cz*m[9]  + m[13];
            const float clipW = cx*m[3] + cy*m[7] + cz*m[11] + m[15];

            const float wMin = clipW - (radius + hdr_.depthSlack) * hdr_.wGradMag;
            if (wMin <= hdr_.nearClipW) return MaskedOcclusionCulling::VISIBLE;

            const float invW = 1.0f / clipW;
            const float cxN = clipX * invW, cyN = clipY * invW;
            const float rxN = radius * hdr_.ndcRadiusX * invW;
            const float ryN = radius * hdr_.ndcRadiusY * invW;

            float minX = cxN - rxN, minY = cyN - ryN;
            float maxX = cxN + rxN, maxY = cyN + ryN;
            if (minX < -1.0f) minX = -1.0f;
            if (minY < -1.0f) minY = -1.0f;
            if (maxX >  1.0f) maxX =  1.0f;
            if (maxY >  1.0f) maxY =  1.0f;
            if (minX >= maxX || minY >= maxY)
                return MaskedOcclusionCulling::VIEW_CULLED;

            return moc_->TestRect(minX, minY, maxX, maxY, wMin);
        }

    private:
        MaskedOcclusionCulling* moc_ = nullptr;
        Header hdr_ = {};
        bool ready_ = false;
        std::int32_t implCreated_ = -1, maskW_ = 0, maskH_ = 0;
    };

} // namespace OcclusionMask
