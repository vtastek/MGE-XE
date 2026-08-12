#pragma once

#include "proxydx/d3d8header.h"

// Forward declaration so accessors can hand back typed scene-graph nodes without
// pulling the MWSE SharedSE NI headers into every consumer of mwbridge.h.
namespace NI { struct Node; struct Camera; }

//-----------------------------------------------------------------------------

class MWBridge {
public:
    ~MWBridge();

    // Singleton access
    static MWBridge* get();

    // Connect to Morrowind memory
    void Load();

    // Used to determine whether we have connected to Morrowind's dynamic memory yet
    inline bool IsLoaded();
    bool CanLoad();

    DWORD GetAlwaysRun();
    DWORD GetAutoRun();
    DWORD GetShadowToggleAddr();
    DWORD GetShadowRealAddr();
    DWORD GetShadowFovAddr();
    DWORD GetCrosshair2();
    void SetCrosshairEnabled(bool enabled);
    void ToggleCrosshair();
    bool IsExterior();
    bool IsMenu();
    bool IsLoadScreen();
    bool IsCombat();
    bool IsCrosshair();
    bool IsAlwaysRun();

    DWORD GetNextTrack();
    DWORD GetMusicVol();
    void SkipToNextTrack();
    void DisableMusic();

    DWORD GetCurrentWeather();
    DWORD GetNextWeather();
    float GetWeatherRatio();
    const RGBVECTOR* getCurrentWeatherSkyCol();
    const RGBVECTOR* getCurrentWeatherFogCol();

    // -- W10: the whole WeatherController, read at once ---------------------------
    // MGE has always poked this struct for two colours (eCurSkyCol/eCurFogCol). The sky
    // takeover needs the rest of it: MW runs ONE global weather state, flips part of it to
    // underwater values the instant the eye submerges, and the sky inherits that — which is
    // why the submerged sky dims in the DX9 baseline too (proven with F11). Reading the whole
    // struct turns "something dims" into a measurement.
    //
    // Every offset below is published in the MWSE headers — TES3::WeatherController
    // (TES3WeatherController.h) and TES3::Weather (TES3Weather.h) — and the two MGE already
    // used (0x90 sky, 0x9c fog) line up exactly, which is the cross-check that the base
    // pointer is right. Nothing here was disassembled (Prime Directive 6).
    //
    // It also settles the standing question about weather continuing while submerged: the
    // particle counts and the thunder-flash intensity are live engine counters, so if they
    // keep moving underwater the simulation never paused and only the LOOK changed.
    struct WeatherState {
        int       curWeather, nextWeather;        // TES3::WeatherType index, -1 if absent
        float     transition;                     // transitionScalar          0x170
        RGBVECTOR skyCol, fogCol;                 // currentSkyColor/FogColor  0x90 / 0x9c
        RGBVECTOR uwCol;                          // underwaterCol             0x1b4
        float     uwWeight;                       // underwaterColWeight       0x1c0
        float     uwFog[5];                       // sunrise/day/sunset/night/indoor 0x1a0..
        float     thunderFlash;                   // activeThunderFlashIntensity 0x178
        int       rainParticles, snowParticles;   //                       0x13c / 0x140
        float     sunglareVis;                    // smoothedSunglareVis       0xd0
        bool      sunOccluded;                    // isSunOccluded             0x1ec
        bool      uwSoundState;                   // currentWeather->underwaterSoundState 0x20d
        // The physical-sky input set: everything a generated sky would need from MW, with
        // all the authored COLOURS demoted to optional tints.
        float     cloudsMaxPercent, cloudsSpeed;  // Weather 0xf0 / 0xfc
        float     landFogDay, landFogNight;       // Weather 0xf4 / 0xf8
        float     windSpeed;                      // Weather 0x100
        const char* cloudTexture;                 // Weather 0x104 (never freed by us)
        // The sky meshes whose per-vertex colours MW rebakes in place each frame. Handed back
        // untyped so this header keeps its "no SharedSE NI headers" promise; the geometry
        // cache casts them (it already owns that idiom).
        void*     triAtmosphere;                  // sgTriAtmosphere      0x7c
        void*     triCloudsCurrent;               // sgTriCloudsCurrent   0x80
    };
    bool getWeatherState(WeatherState& out);

    // -- R0: MW's live RIPPLE POOL ------------------------------------------------
    // TES3::WaterController keeps a fixed pool of ripple TriShapes (MaxNumberRipples=75 in
    // Morrowind.ini) and drives each one from an alpha controller over RippleLifetime. The
    // engine spawns them itself for anything moving in water, which is the whole point: MGE's
    // DX9 ripple was ONE origin (the player) because that was all the shader had, while MW has
    // always had the full list. Reading the pool restores NPC and creature ripples without us
    // detecting a single actor.
    //
    // Offsets are MWSE-published: TES3::WaterController / ::Ripple (TES3WaterController.h) and
    // NI::TimeController (SharedSE/NITimeController.h). The base cross-checks against MGE's own
    // long-standing markWaterNode(), which already walks dataHandler+0xB4EC -> +0xB4 (waterPlane).
    // Nothing disassembled (Prime Directive 6).
    //
    // ⚠ The AGE comes from the alpha controller, not from a scale we reverse-engineer. Both the
    // TriShape scale and the controller advance on the same clock, but only the controller states
    // its own range (low/highKeyFrame), so deriving age from it needs no assumption about whether
    // MW's growth curve is linear — and RippleScale=0.15,6.5 is exactly the kind of pair that
    // invites that assumption.
    struct RippleSource {
        float x, y;        // world XY of the ripple centre (TriShape world translation)
        float age;         // 0 = just spawned, 1 = at the end of its life
        float scale;       // TriShape world scale, i.e. the ring's current radius factor
    };
    struct RippleState {
        int   count;               // entries written into `out`
        int   poolSize;            // maxRippleCount            0x50
        int   activeInPool;        // isActive entries, even those we could not read
        float lifetime;            // rippleLifetime            0x34
        float scaleBegin, scaleEnd;// rippleScaleX / Y          0x38 / 0x3c
        float alphas[3];           // rippleAlphas[3]           0x40
        float rotSpeed;            // rippleRotationSpeed       0x4c
        const char* rippleTexture; // rippleTexturePath         0x08 (never freed by us)
        void* rippleNode;          // NI::Node parent           0xb0 — the SUPPRESSION handle
    };
    // Fills up to `max` sources; returns false when there is no water controller (no loaded game).
    bool getRippleState(RippleState& out, RippleSource* sources, int max);

    DWORD getScenegraphFogCol();
    void setScenegraphFogCol(DWORD c);
    float getScenegraphFogDensity();
    bool CellHasWeather();
    float* GetWindVector();
    DWORD GetWthrStruct(int wthr);
    int GetWthrString(int wthr, int offset, char str[]);
    void SetWthrString(int wthr, int offset, char str[]);
    bool CellHasWater();
    bool IsUnderwater(float eyeZ);
    bool WaterReflects(float eyeZ);
    float simulationTime();
    float frameTime();

    float* getMouseSensitivityYX();
    float GetViewDistance();
    void SetViewDistance(float dist);
    float GetAIDistance();
    void SetAIDistance(float dist);

    void SetFOV(float screenFOV);

    void GetSunDir(float& x, float& y, float& z);
    BYTE GetSunVis();
    void setSunriseSunset(float rise_time, float rise_dur, float set_time, float set_dur);

    DWORD IntCurCellAddr();
    bool IntLikeExterior();
    bool IntIllegSleep();
    bool IntHasWater();
    float WaterLevel();

    const char* getInteriorName();
    const BYTE* getInteriorAmb();
    const BYTE* getInteriorSun();
    const BYTE* getInteriorFog();
    float getInteriorFogDens();

    DWORD PlayerPositionPointer();
    float PlayerPositionX();
    float PlayerPositionY();
    float PlayerPositionZ();
    float PlayerHeight();
    bool IsPlayerWaiting();
    D3DXVECTOR3* PCam3Offset();
    DWORD getPlayerMACP();
    bool is3rdPerson();
    // The player reference's scene-graph node = the 3rd-person body (the engine
    // appCulls it while in 1st person). Null during load screens / before MACP exists.
    NI::Node* getPlayer3rdPersonNode();
    // FP1a: the WorldController armCamera scene root — the first-person arms/weapon
    // subtree MW renders in its own post-z-clear scene. Null when unavailable.
    NI::Node* getArmCameraRoot();
    // FP1c: the armCamera's NiCamera (CameraData::camera at wc+0x150+0x10), typed as
    // NI::Camera*. Used to re-face first-person billboards (held candle/enchant glow)
    // toward the arm view during the FP capture walk — MW's own billboard re-orient runs
    // in the arm cull/render pass that FP suppression skips, so the facing goes stale.
    // Null when unavailable.
    NI::Camera* getArmCamera();
    // MW-ONLY-UI: the WORLD camera's NiCamera (the which==0 counterpart of getArmCamera).
    // Needed to re-face world billboards ourselves once suppression stops MW traversing the
    // world roots — MW re-orients NiBillboardNode during its cull pass, so without that pass
    // flame/glow quads keep last frame's facing (edge-on, and tilted off vertical). Null when
    // unavailable.
    NI::Camera* getWorldCamera();
    // FP1a: world-space state of one WorldControllerRenderCamera (which: 0 =
    // worldCamera/main view, 1 = armCamera/first person): the NiCamera basis
    // (pos/dir/up/right) + the CameraData projection params {fovDegrees (HORIZONTAL),
    // nearPlane, farPlane, viewportW, viewportH}. MW builds its D3D projection from
    // CameraData, NOT from the NiCamera's Gamebryo viewFrustum — validated in-game:
    // the frustum carries a different (MGE-patched) FOV and near=1 while the D3D proj
    // follows CameraData (fov 75°, near 4). Returns false when unresolvable.
    bool getRenderCameraState(int which, float pos[3], float dir[3], float up[3],
                              float right[3], float camData[5]);
    // FP cam diag: the raw Gamebryo viewFrustum {left,right,top,bottom,near,far} +
    // viewport port {l,r,t,b} of a WorldControllerRenderCamera's NiCamera (which: 0 =
    // worldCamera, 1 = armCamera). Cull state, NOT the projection authority (see above) —
    // logged for triage only.
    bool getRenderCameraFrustum(int which, float frustum[6], float port[4]);
    // Live scene-graph sunlight (TES3DataHandler+0x98, NI::DirectionalLight): the
    // authoritative light MW programs D3D light 6 FROM, interior AND exterior — fresh
    // every frame regardless of whether MW re-sent any D3D light state. dir is the
    // world-space travel direction (unnormalized); colors are pre-dimmer.
    bool getSceneSunlight(float dir[3], float diffuse[3], float ambient[3], float* dimmer);
    DWORD getPlayerTarget();
    int getPlayerWeapon();
    bool isPlayerCasting();
    bool isPlayerAimingWeapon();
    void* getPlayerCell();

    void HaggleMore(DWORD num);
    void HaggleLess(DWORD num);

    void toggleRipples(BOOL enabled);
    void markWaterNode(float k);
    void markMoonNodes(float k);
    // Returns the two moon root scene-graph nodes (Masser, Secunda), either null when
    // unavailable. Reuses the markMoonNodes offset chain (eMaster -> weather controller
    // -> masser/secunda -> moon root). Each root parents the moon's Shadow Node cutout
    // and Moon Node disc billboards; the engine's per-node appCulled flag encodes "is
    // this moon up". Used by the water reflection to draw moons independently of the
    // main-camera frustum (recordSky only captures moons the main view actually drew).
    void getMoonRootNodes(NI::Node** masser, NI::Node** secunda);
    void disableScreenshotFunc();
    void disableSunglare();
    void disableIntroMovies();
    bool isIntroDone();
    bool isLoadingBar();
    void showLoadingBar(const char* text, float amount);

    HWND getWindowHandle();
    void* getGameOptionsStruct();
    void destroyLoadingBar();
    void patchGameLoading(void (__cdecl* newfunc)());
    void redirectMenuBackground(void (__stdcall* func)(int));
    float getUIScale();
    void setUIScale(float scale);
    void patchUIConfigure(void (__stdcall* newfunc)());
    void patchSplashScreen(unsigned int width, unsigned int height);
    void patchFrameTimer(int (__cdecl* newfunc)());
    void patchResolveDuringInit(void (__cdecl* newfunc)());
    void patchLoadTexture2D();
    void patchLightParticleMaterialModifier();
    void patchWorldRenderingAccumulation();

    void* getGMSTPointer(DWORD id);
    DWORD getKeybindCode(DWORD action);
    const char* getPlayerName();
    float getGameHour();
    int getDaysPassed();
    int getFrameBeginMillis();
    void* getGlobalVar(const char *id);
    float getGlobalVarValue(const void* globalVar);
    void* getDialogue(const char *id);
    int getJournalIndex(const void* dialogue);
    void* findFirstReferenceById(const char *id);
    unsigned int getRecordFlags(const void* record);

    MWBridge();

protected:
    DWORD m_version;
    bool m_loaded;

    /// Sets pointers to static memory of Morrowind
    void InitStaticMemory();

    /// Functions for reading and writing data at locations in Morrowind's memory
    DWORD read_dword(const DWORD dwAddress);
    WORD read_word(const DWORD dwAddress);
    BYTE read_byte(const DWORD dwAddress);
    float read_float(const DWORD dwAddress);
    void write_dword(const DWORD dwAddress, DWORD dword);
    void write_word(const DWORD dwAddress, WORD word);
    void write_byte(const DWORD dwAddress, BYTE byte);
    void write_float(const DWORD dwAddress, float f);
    void write_ptr(const DWORD dwAddress, void* ptr);

    /// Pointers to Morrowind Memory
    DWORD
    eMaster, eEnviro, eMaster1, eMaster2,
             eFPS, eTimer, eD3D, eTruRenderWidth, eShadowSlider,
             eCrosshair1, eAI, eView0, eRenderWidth,
             eView1, eCombat, ePCRef,

             eGamma, eView4, eLookMenu,

             eX, eCos, eWorldFOV, eView2,

             eSkyFOV, eMenuFOV, eView3, eExt, eMenu, eMouseLim,

             eLoad,

             eWthrArray, eCurWthrStruct, eNextWthrStruct,
             eCurSkyCol, eCurFogCol,
             eWindVector,
             eSunriseHour, eSunsetHour, eSunriseDuration, eSunsetDuration,
             eSunDir, eSunVis, // Real sun direction, sun(glare) alpha value
             eWeatherRatio;

    // floating point variables
    DWORD eNextTrack, eMusicVol,
          eAlwaysRun, eAutoRun,
          eShadowToggle, eShadowReal, eShadowFOV,
          eCrosshair2;

    // Pointers to Morrowind code
    DWORD eNoMusicBreak,
          eGammaFunc,
          eMusicVolFunc,
          eHaggleUpdate, eHaggleAmount,
          eMenuMouseMove,
          eTruform, eGetMouseState,
          eXMenuHudIn, eXMenuHudOut, eXMenuNoMouse, eXMenuNoFOV,
          eXMenuWnds, eXMenuPopups, eXMenuLoWnds, eXMenuSubtitles, eXMenuFPS,
          eNoWorldFOV, eXRotSpeed, eYRotSpeed,
          eScrollScale, eBookScale, eJournalScale, eRipplesSwitch;

    // Other values
    DWORD dwAlwaysRunOffset;
};

//-----------------------------------------------------------------------------
// Inline Functions
//-----------------------------------------------------------------------------

inline bool MWBridge::IsLoaded() {
    return m_loaded;
}

//-----------------------------------------------------------------------------
