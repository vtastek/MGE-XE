#pragma once

#include <basetyps.h>



#define MASK(x) (1U << x)

// MGE Generic flags
#define MGE_DISABLED_BIT        0
#define MGE_DISABLED            MASK(MGE_DISABLED_BIT)
#define FOG_ENABLED_BIT         1
#define FOG_ENABLED             MASK(FOG_ENABLED_BIT)
#define FPS_COUNTER_BIT         2
#define FPS_COUNTER             MASK(FPS_COUNTER_BIT)
#define DISPLAY_MESSAGES_BIT    3
#define DISPLAY_MESSAGES        MASK(DISPLAY_MESSAGES_BIT)
#define USE_HW_SHADER_BIT       4
#define USE_HW_SHADER           MASK(USE_HW_SHADER_BIT)
#define NO_MW_SUNGLARE_BIT      5
#define NO_MW_SUNGLARE          MASK(NO_MW_SUNGLARE_BIT)
#define INPUT_LAG_FIX_BIT       6
#define INPUT_LAG_FIX           MASK(INPUT_LAG_FIX_BIT)
#define USE_MENU_CACHING_BIT    7
#define USE_MENU_CACHING        MASK(USE_MENU_CACHING_BIT)
#define ZOOM_ASPECT_BIT         8
#define ZOOM_ASPECT             MASK(ZOOM_ASPECT_BIT)
#define MWSE_DISABLED_BIT       9
#define MWSE_DISABLED           MASK(MWSE_DISABLED_BIT)
#define USE_FFESHADER_BIT       10
#define USE_FFESHADER           MASK(USE_FFESHADER_BIT)
#define TRANSPARENCY_AA_BIT     11
#define TRANSPARENCY_AA         MASK(TRANSPARENCY_AA_BIT)
#define USE_HDR_BIT             12
#define USE_HDR                 MASK(USE_HDR_BIT)
#define CROSSHAIR_AUTOHIDE_BIT  13
#define CROSSHAIR_AUTOHIDE      MASK(CROSSHAIR_AUTOHIDE_BIT)
#define SKIP_INTRO_BIT          14
#define SKIP_INTRO              MASK(SKIP_INTRO_BIT)
#define CPU_IDLE_BIT            15
#define CPU_IDLE                MASK(CPU_IDLE_BIT)
// Distant Land flags
#define USE_DISTANT_WATER_BIT   16
#define USE_DISTANT_WATER       MASK(USE_DISTANT_WATER_BIT)
#define USE_DISTANT_LAND_BIT    17
#define USE_DISTANT_LAND        MASK(USE_DISTANT_LAND_BIT)
#define USE_DISTANT_STATICS_BIT 18
#define USE_DISTANT_STATICS     MASK(USE_DISTANT_STATICS_BIT)
#define NO_INTERIOR_DL_BIT      19
#define NO_INTERIOR_DL          MASK(NO_INTERIOR_DL_BIT)
#define REFLECTIVE_WATER_BIT    20
#define REFLECTIVE_WATER        MASK(REFLECTIVE_WATER_BIT)
#define REFLECT_NEAR_BIT        21
#define REFLECT_NEAR            MASK(REFLECT_NEAR_BIT)
#define REFLECT_INTERIOR_BIT    22
#define REFLECT_INTERIOR        MASK(REFLECT_INTERIOR_BIT)
#define NOT_USING_DL_BIT        23
#define NOT_USING_DL            MASK(NOT_USING_DL_BIT)
#define NO_MW_MGE_BLEND_BIT     24
#define NO_MW_MGE_BLEND         MASK(NO_MW_MGE_BLEND_BIT)
#define REFLECT_SKY_BIT         25
#define REFLECT_SKY             MASK(REFLECT_SKY_BIT)
#define DYNAMIC_RIPPLES_BIT     26
#define DYNAMIC_RIPPLES         MASK(DYNAMIC_RIPPLES_BIT)
#define BLUR_REFLECTIONS_BIT    27
#define BLUR_REFLECTIONS        MASK(BLUR_REFLECTIONS_BIT)
#define EXP_FOG_BIT             28
#define EXP_FOG                 MASK(EXP_FOG_BIT)

#define USE_ATM_SCATTER_BIT     29
#define USE_ATM_SCATTER         MASK(USE_ATM_SCATTER_BIT)
#define USE_GRASS_BIT           30
#define USE_GRASS               MASK(USE_GRASS_BIT)
#define USE_SHADOWS_BIT         31
#define USE_SHADOWS             MASK(USE_SHADOWS_BIT)



typedef unsigned long DWORD;
typedef unsigned char BYTE;
#pragma once

struct ConfigurationStruct {
    DWORD MGEFlags;
    bool OnlyProxyD3D8To9;
    BYTE AALevel;
    BYTE ZBufFormat;
    BYTE VWait;
    BYTE RefreshRate;
    // MGE-owned frame limiter target (fps). 0 = off (uncapped). Read from
    // MGE.ini; the engine's own "Max FPS" limiter is neutralized by MGEgui
    // forcing Morrowind.ini Max FPS high, so this is the sole pacer.
    int FPSLimit;
    bool Borderless;
    BYTE AnisoLevel;
    BYTE ScaleFilter;
    bool UseDefaultTexturePool;
    float ScreenFOV;
    BYTE FogMode;
    BYTE SSFormat;
    BYTE SSSuffix;
    char SSDir[208];
    char SSName[32];
    float HDRReactionSpeed;
    DWORD PerPixelLightFlags;
    int StatusTimeout;
    bool Force3rdPerson;
    struct {
        float x, y, z;
    } Offset3rdPerson;
    float UIScale;
    int WindowAlignX, WindowAlignY;
    bool UseSharedMemory;
    bool UseOcclusionCulling;  // reuse msoc.dll's CPU occlusion mask for distant statics
    bool UseHostOcclusionCull; // ship the mask to the 64-bit host so it occlusion-culls distant statics in the quadtree walk (only survivors cross IPC). Default off.
    int OcclusionHysteresisFrames;  // consecutive OCCLUDED frames before a static actually culls
    float OcclusionSphereInflate;   // per-instance sphere/OBB radius scale for verdict stability
    bool LogDistantPipeline;        // gate per-frame diagnostic loglines + phase-timer reports + Numpad-5 mask dump
    bool UseSceneGraphSnapshot;     // enable MGE-side per-frame scene-graph walk (drives the texture-light variant of FFE)
    bool UseAsyncSceneGraphWalk;    // sub-flag: run the scene-graph walk on a worker thread; main signals at onFrameReady and returns immediately. Snapshot is one frame stale. Default off; on hides the ~350µs walk from the main-thread frame budget.
    bool UseRenderThread;           // enable the MGE render thread: submit GPU work on a second core during the engine's CPU-only frame-start/sky windows. Also forces D3DCREATE_MULTITHREADED and arms the device-submission lock. Default off.
    bool UseTiledLights;            // main-view point lights via per-frame screen-tile binning (USE_TILED_LIGHTS) instead of per-mesh selection. Rides the scene-graph snapshot. Runtime A/B on VK_DECIMAL. Default off.
    bool UseWaterFlowMap;           // per-water-body directional waves & storm-calm ponds: bake a low-res flow map (distance-to-sea BFS over the wet/dry grid) and sample it in the water shader (WATER_FLOW_MAP). Runtime A/B on VK_NUMPAD9. Default off. (MGEFlags bits 0-31 are full, so this is a standalone bool like UseTiledLights.)
    bool UseRenderProcess;          // present-seam spike: bring up the out-of-process 64-bit Vulkan renderer (mgeHost64) and composite its output into MW's window. Gates RenderInit + framebuffer-vec setup; per-frame blit additionally toggled by a debug key (F11). Default off; off = game byte-for-byte unchanged. (Milestone A — CPU readback — works on the plain D3D9 device.)
    bool UseAsyncHostFrame;         // async client↔host frame split: kick off the Forge host RenderFrame early and finish (wait+composite) at the scene-0 composite point, overlapping the host render with MW's own frame work. Off = fused kickoff+finish at the composite point (the exact pre-split serial behaviour) for A/B. Only meaningful with UseRenderProcess. Default on.
    bool ForgeFrameAhead;           // pipeline a frame ahead (deferred finish): on early-kickoff frames the paired renderSceneFinish + RT copy move to the NEXT frame's BeginScene(0) collect — before frameSetupEarly, so the IPC window closes before any of the new frame's RPCs — and the EndScene(0) composite point blits the PREVIOUS host frame from g_mainTex with zero IPC/wait. The host D3D12 frame overlaps the WHOLE MW frame (scene 1, UI, present, sim) instead of just scene 0. Composited world lags input by one frame (UI stays current); screenshots capture the 1-frame-old world. Seeds the numpad-* live A/B toggle. Off = same-frame finish, byte-identical to before. Only meaningful with UseAsyncHostFrame. Default on.
    bool ForgeNearDepthReplay;      // near-scene no-op A/B: when Forge owns the opaque world, redraw the near cache DEPTH-ONLY into the main depthstencil (renderCacheDepthToMainZ, ~3.5k draws/frame) so scene-1 sorted-alpha occludes against the suppressed opaques. Off (default) = skip the replay entirely — sorted-alpha may show through Forge walls until the Forge alpha pass lands; the composite overwrite already hides most cases.
    bool ForgeLiveDrawBuild;        // W3: while Forge owns the opaque world, skip the per-frame GeomCache refresh walk entirely — buildFrustumVisibleSet freshens exactly the classify-visible keys straight off their live NiTriShapes (ensureLive: transform/palette/mirrored + revision-gated re-upload + lazy capture on first sight); frames with no classify fall back to a full walk. Eviction switches to a pure age rule on live frames (off-screen != gone). Off = full refresh walk every frame (W1.5 behavior).
    bool ForgeActiveCellWalk;       // W1.5: while Forge owns the opaque world (exteriors), the GeomCache walk skips whole cell subtrees whose world bound lies beyond MW's view distance (far-corner corrected) — the engine can't draw anything there, and the Forge host renders the far world from its own data. Skipped entries are kept by eviction hysteresis so returning doesn't re-capture/re-upload. Off = full walk every frame (pre-W1.5 behavior).
    bool ForgeOpaqueDisplaySkip;    // engine scene-0 opaque no-op via msoc.dll: while Forge owns the opaque world, push mwse_setOpaqueWorldOwned so the plugin's deferred-display drain SKIPS engine display() of covered-opaque leaves (the proxy rejects their DIPs anyway) — removes the traversal/state/DIP cost of ~3.2k dead draws. Alpha-blended/decal/untextured leaves keep displaying (scene-1 sorted alpha unaffected). Needs msoc.dll with the export + MSOC active in the scene; degrades to today's reject-per-DIP path otherwise. Default on; F11 off restores full engine display at runtime.
    bool ForgeAlphaPass;            // AT1 sorted-alpha takeover: capture the scene-1 alpha-BLENDED world shapes (banners/tapestries/foliage/glass) at kickoff, ship them back-to-front sorted, and draw them in the Forge host after water (depth GEQUAL test, no write, cull NONE) so they occlude correctly behind Forge walls. One gate for capture + emit + host draw. Default on (in-game verified 2026-07-03).
    bool ForgeAlphaSuppressS1;      // AT1 suppression: while the Forge alpha pass is live (ForgeAlphaPass + forgeOwnsFrame), reject MW's BLENDED non-water DIPs in scenes >= 1 in inspectIndexedPrimitive so the blended set renders ONCE (host only). Blended-only, NOT a scene-index gate — MW scene indices are conditional (no sorted alpha => scene 1 IS 1st person; hands must keep drawing). OFF = double-draw A/B; what still shows with it ON is the AT3 leftover set (particles/VFX — not NiTriShapes, not captured). Default on.
    bool ForgeAlphaCapture;         // AT3 captured-alpha: at the ForgeAlphaSuppressS1 reject gate, Lock the live blended DIP's VB/IB, copy its final billboarded verts, and ship them next-frame to the host to draw in the same post-water sorted-alpha pass — restores NiParticles (chimney smoke, candle/camp flames) and multimap/decal/untextured blends the host cache pass doesn't own. Pure MGE-XE (client capture + host draw); no msoc plugin change. Set 0 for old-msoc-dll runs (opaque-only skip → cached single-map blends would double-draw). Default on.
    bool ForgeFPPass;               // FP1a first-person takeover: walk the engine's arm-scene root (WorldController armCamera) each 1st-person frame, ship the arms/weapon draws + the arm camera's own view/proj to the Forge host, and draw them there in a dedicated FP pass (fresh reverse-Z depth, world screen-space AO/shadow masks neutralized). MW's own FP draws still run unless ForgeFPSuppress. Default off.
    bool ForgeFPSuppress;           // FP1b: while the Forge FP pass is live (ForgeFPPass + forgeOwnsFrame + 1st person), force the engine's arm-scene root appCulled each frame so MW's own first-person arms never draw (no double image). Restored on any gate release (F11 off / 3rd person / host dead). Dev-key toggle flips this live for A/B. Default off.
    bool ForgeEmissiveBoost;        // boost a light fixture's emissive from its own light's colour. A fixture's emissive texture is its light colour HDR-clipped (tex ~= clip(k*L), k ~= 4.5), so the authored emission is ~4.5x the light and an 8-bit texture can't hold it — lanterns read pale cream instead of hot orange. Recovers the clipped channels with the per-channel gain max(k*L,1), folded into matEmissive at draw-list build (Forge path only; the DX9 baseline stays vanilla for A/B). Live toggle. Default on.
    bool UseRenderProcessEx;        // present-seam spike Milestone B: upgrade MW's device to D3D9Ex (for a shared render-target HANDLE → zero-copy Vulkan). SEPARATE from UseRenderProcess so the working A path stays on plain D3D9. NOTE: under DXVK, Ex windowed flip-present + MGE borderless = VK_ERROR_SURFACE_LOST_KHR (white screen) — under investigation. Default off.

    struct {
        float zoom, zoomRate, zoomRateTarget;
        bool rotateUpdate;
        float rotation, rotationRate;
        bool shake;
        float shakeMagnitude, shakeAccel;
    } CameraEffects;

    struct {
        float DrawDist;
        float NearStaticEnd;
        float FarStaticEnd;
        float VeryFarStaticEnd;
        float FarStaticMinSize;
        float VeryFarStaticMinSize;
        float AboveWaterFogStart;
        float AboveWaterFogEnd;
        float BelowWaterFogStart;
        float BelowWaterFogEnd;
        float InteriorFogStart;
        float InteriorFogEnd;
        BYTE WaterWaveHeight;
        BYTE WaterCaustics;
        DWORD ShadowResolution;
        float ShadowNearStaticRadius;   // include static objects within this distance in near cascade (0 = off)
        float Wind[10];
        float FogD[10];
        float FgOD[10];
    } DL;

    struct {
        float SunMult[10];
        float AmbMult[10];
    } Lighting;

    struct {
        bool AltCombat;
        char Macros[4096];
        char Triggers[4096];
        char Remap[4096];
    } Input;

    char ShaderChain[512];

    bool LoadSettings();
    bool SaveSettings();
};

extern ConfigurationStruct Configuration;
