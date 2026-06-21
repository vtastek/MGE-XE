#pragma once

// Minimal, self-contained declarations of DXVK's D3D9<->Vulkan interop interfaces.
//
// DXVK's own src/d3d9/d3d9_interfaces.h pulls DXVK-internal headers (d3d9_include.h,
// vulkan_loader.h) and cannot be included here. These declarations are copied VERBATIM
// from that header (same IIDs, same method order) so the COM vtable layout matches DXVK
// exactly — the dispatch is by vtable slot, so order is load-bearing. Backed by the
// official Vulkan SDK headers rather than DXVK's loader.
//
// Obtain ID3D9VkInteropDevice by QueryInterface on MW's (DXVK) IDirect3DDevice9; obtain
// ID3D9VkInteropTexture by QueryInterface on a DXVK IDirect3DTexture9 (or its surface).

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <d3d9.h>
#include <cstdint>

// --- D3D9VkExtImageDesc (d3d9_interfaces.h) ---------------------------------------------
struct D3D9VkExtImageDesc {
    D3DRESOURCETYPE     Type;               // SURFACE, TEXTURE, CUBETEXTURE, VOLUMETEXTURE
    UINT                Width;
    UINT                Height;
    UINT                Depth;              // > 1 for VOLUMETEXTURE
    UINT                MipLevels;          // > 1 for TEXTURE/CUBETEXTURE/VOLUMETEXTURE
    DWORD               Usage;
    D3DFORMAT           Format;
    D3DPOOL             Pool;
    D3DMULTISAMPLE_TYPE MultiSample;        // NONE unless Type is SURFACE
    DWORD               MultiSampleQuality;
    bool                Discard;            // depth stencils only
    bool                IsAttachmentOnly;   // if false, VK_IMAGE_USAGE_SAMPLED_BIT is added
    bool                IsLockable;
    VkImageUsageFlags   ImageUsage;         // additional image usage flags
};

// --- ID3D9VkInteropTexture (IID d56344f5-8d35-46fd-806d-94c351b472c1) -------------------
MIDL_INTERFACE("d56344f5-8d35-46fd-806d-94c351b472c1")
ID3D9VkInteropTexture : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE GetVulkanImageInfo(
            VkImage*              pHandle,
            VkImageLayout*        pLayout,
            VkImageCreateInfo*    pInfo) = 0;
};

// --- ID3D9VkInteropDevice (IID 2eaa4b89-0107-4bdb-87f7-0f541c493ce0) --------------------
MIDL_INTERFACE("2eaa4b89-0107-4bdb-87f7-0f541c493ce0")
ID3D9VkInteropDevice : public IUnknown {
    virtual void STDMETHODCALLTYPE GetVulkanHandles(
            VkInstance*           pInstance,
            VkPhysicalDevice*     pPhysDev,
            VkDevice*             pDevice) = 0;

    virtual void STDMETHODCALLTYPE GetSubmissionQueue(
            VkQueue*              pQueue,
            uint32_t*             pQueueIndex,
            uint32_t*             pQueueFamilyIndex) = 0;

    virtual void STDMETHODCALLTYPE TransitionTextureLayout(
            ID3D9VkInteropTexture*    pTexture,
      const VkImageSubresourceRange*  pSubresources,
            VkImageLayout             OldLayout,
            VkImageLayout             NewLayout) = 0;

    virtual void STDMETHODCALLTYPE FlushRenderingCommands() = 0;

    virtual void STDMETHODCALLTYPE LockSubmissionQueue() = 0;

    virtual void STDMETHODCALLTYPE ReleaseSubmissionQueue() = 0;

    virtual void STDMETHODCALLTYPE LockDevice() = 0;

    virtual void STDMETHODCALLTYPE UnlockDevice() = 0;

    virtual bool STDMETHODCALLTYPE WaitForResource(
            IDirect3DResource9*  pResource,
            DWORD                MapFlags) = 0;

    virtual HRESULT STDMETHODCALLTYPE CreateImage(
            const D3D9VkExtImageDesc* desc,
            IDirect3DResource9**      ppResult) = 0;
};
