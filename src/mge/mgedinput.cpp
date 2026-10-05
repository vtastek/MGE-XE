
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <windows.h>
#include <mmsystem.h>
#include "support/log.h"
#include "mgedinput.h"
#include "configuration.h"
#include "mgeversion.h"
#include "mmefunctiondefs.h"
#include "mwbridge.h"


bool MGEProxyDirectInput::mouseClick = false;
int MGEProxyDirectInput::modifierKeys = 0;


typedef void (*FakeFunc)();

static FakeFunc FakeFuncs[MGEINPUT_GRAPHICSFUNCS];  // See GraphicsFuncs enum
static sFakeKey FakeKeys[MGEINPUT_MAXMACROS];       // Last 10 reserved for mouse
static sFakeTrigger Triggers[MGEINPUT_MAXTRIGGERS]; // Up to 4 time delayed triggers
static BYTE RemappedKeys[256];

// Input device macro variables
static BYTE LastBytes[MGEINPUT_MAXMACROS];          // Which keys were pressed last GetData call
static BYTE FakeStates[MGEINPUT_MAXMACROS];         // Which keys are currently permanently down
static BYTE HammerStates[MGEINPUT_MAXMACROS];       // Which keys are currently being hammered
static BYTE AHammerStates[MGEINPUT_MAXMACROS];      // Which keys are currently being ahammered
static BYTE TapStates[MGEINPUT_MAXMACROS];          // Which keys need to be tapped next frame
static BYTE DisallowMask[MGEINPUT_MAXMACROS];       // Mask of which keys are disallowed
static DIDEVICEOBJECTDATA FakeBuffer[256];          // Stores the list of fake keypresses to send to console
static DWORD FakeBufferStart;                       // The index of the next character to write from FakeBuffer[]
static DWORD FakeBufferEnd;                         // The index of the last character contained in FakeBuffer[]
static DWORD TriggerFireTimes[MGEINPUT_MAXTRIGGERS];
static bool FinishedFake;                           // true to shut down the console
static bool CloseConsole;                           // true to shut the console after performing a command
static BYTE MouseIn[10];                            // Used to transfer keypresses to the mouse
static BYTE MouseOut[10];                           // Used to transfer keypresses back from the mouse

enum AttackState { State_NONE = 0, State_SLASH, State_PIERCE, State_CHOP, State_NOMOTION  };

static struct {
    bool directionPressed;          // True when the player makes an attack (Because the keyboard must be used before the mouse)
    bool directionPressedLast;
    int attackType;                 // Used to store the state of the mouse between frames
    int lastAttack;                 // You cant use the same attack direction twice in a row
} AltCombat = {
    false, false, 0, 0
};

const int altSensitivity = 18;      // How many pixels the mouse must move to register an attack
const int maxGap = 15;              // If the difference between x and y movement is greater than this, don't use a hacking attack

static bool GlobalHammer;



static void loadInputSettings();
static void stub() {}

void* CreateInputWrapper(void* real) {
    // Read macros and triggers
    loadInputSettings();

    // Initial state
    GlobalHammer = true;       // Used to hammer keys (alternates between up/down)

    FakeBufferStart = 0;
    FakeBufferEnd = 0;
    FinishedFake = false;      // true to shut down the console
    CloseConsole = false;      // true to shut the console after performing a command

    ZeroMemory(&LastBytes, sizeof(LastBytes));
    memset(DisallowMask, 0xff, sizeof(DisallowMask));

    for (int i = 0; i != MGEINPUT_MAXTRIGGERS; ++i) {
        TriggerFireTimes[i] = GetTickCount() + Triggers[i].TimeInterval;
    }

    // Initialize the array of macro function pointers
    for (int i = 0; i != MGEINPUT_GRAPHICSFUNCS; ++i) {
        FakeFuncs[i] = stub;
    }

    FakeFuncs[GF_Screenshot] = MacroFunctions::TakeScreenshot;
    FakeFuncs[GF_ToggleZoom] = MacroFunctions::ToggleZoom;
    FakeFuncs[GF_IncreaseZoom] = MacroFunctions::IncreaseZoom;
    FakeFuncs[GF_DecreaseZoom] = MacroFunctions::DecreaseZoom;
    FakeFuncs[GF_ResetEnableZoom] = MacroFunctions::ResetEnableZoom;
    FakeFuncs[GF_ToggleText] = MacroFunctions::ToggleStatusText;
    FakeFuncs[GF_ShowLastText] = MacroFunctions::ShowLastMessage;
    FakeFuncs[GF_ToggleFps] = MacroFunctions::ToggleFpsCounter;
    FakeFuncs[GF_IncreaseView] = MacroFunctions::IncreaseViewRange;
    FakeFuncs[GF_DecreaseView] = MacroFunctions::DecreaseViewRange;
    FakeFuncs[GF_ToggleCrosshair] = MacroFunctions::ToggleCrosshair;
    FakeFuncs[GF_NextTrack] = MacroFunctions::NextTrack;
    FakeFuncs[GF_DisableMusic] = MacroFunctions::DisableMusic;
    FakeFuncs[GF_IncreaseFOV] = MacroFunctions::IncreaseFOV;
    FakeFuncs[GF_DecreaseFOV] = MacroFunctions::DecreaseFOV;

    FakeFuncs[GF_HaggleMore1] = MacroFunctions::HaggleMore1;
    FakeFuncs[GF_HaggleMore10] = MacroFunctions::HaggleMore10;
    FakeFuncs[GF_HaggleMore100] = MacroFunctions::HaggleMore100;
    FakeFuncs[GF_HaggleMore1000] = MacroFunctions::HaggleMore1000;
    FakeFuncs[GF_HaggleMore10000] = MacroFunctions::HaggleMore10000;
    FakeFuncs[GF_HaggleLess1] = MacroFunctions::HaggleLess1;
    FakeFuncs[GF_HaggleLess10] = MacroFunctions::HaggleLess10;
    FakeFuncs[GF_HaggleLess100] = MacroFunctions::HaggleLess100;
    FakeFuncs[GF_HaggleLess1000] = MacroFunctions::HaggleLess1000;
    FakeFuncs[GF_HaggleLess10000] = MacroFunctions::HaggleLess10000;

    FakeFuncs[GF_Shader] = MacroFunctions::ToggleShaders;
    FakeFuncs[GF_ToggleDL] = MacroFunctions::ToggleDistantLand;
    FakeFuncs[GF_ToggleShadows] = MacroFunctions::ToggleShadows;
    FakeFuncs[GF_ToggleGrass] = MacroFunctions::ToggleGrass;
    FakeFuncs[GF_ToggleMwMgeBlending] = MacroFunctions::ToggleBlending;
    FakeFuncs[GF_ToggleLightingMode] = MacroFunctions::ToggleLightingMode;
    FakeFuncs[GF_ToggleTrAA] = MacroFunctions::ToggleTransparencyAA;

    FakeFuncs[GF_MoveForward3PC] = MacroFunctions::MoveForward3PCam;
    FakeFuncs[GF_MoveBack3PC] = MacroFunctions::MoveBack3PCam;
    FakeFuncs[GF_MoveLeft3PC] = MacroFunctions::MoveLeft3PCam;
    FakeFuncs[GF_MoveRight3PC] = MacroFunctions::MoveRight3PCam;
    FakeFuncs[GF_MoveDown3PC] = MacroFunctions::MoveDown3PCam;
    FakeFuncs[GF_MoveUp3PC] = MacroFunctions::MoveUp3PCam;

    // Force screenshots from PrintScreen
    FakeKeys[DIK_SYSRQ].type = MT_Graphics;
    FakeKeys[DIK_SYSRQ].Function.index = GF_Screenshot;

    return new MGEProxyDirectInput((IDirectInput8A*)real);
}

void MGEProxyDirectInput::changeKeyBehavior(DWORD key, MGEProxyDirectInput::KeyBehavior kb, bool on) {
    if (key >= MGEINPUT_MAXMACROS) {
        return;
    }

    switch (kb) {
    case MGEProxyDirectInput::TAP:
        TapStates[key] = on ? 0x80 : 0;
        break;
    case MGEProxyDirectInput::PUSH:
        FakeStates[key] = on ? 0x80 : 0;
        break;
    case MGEProxyDirectInput::HAMMER:
        HammerStates[key] = on ? 0x80 : 0;
        break;
    case MGEProxyDirectInput::AHAMMER:
        AHammerStates[key] = on ? 0x80 : 0;
        break;
    case MGEProxyDirectInput::DISALLOW:
        DisallowMask[key] = on ? 0 : 0xff;
        break;
    }
}

static void FakeString(BYTE chars[], BYTE data[], BYTE length) {
    for (int i = 0; i != length; ++i) {
        FakeBuffer[FakeBufferEnd].dwOfs = chars[i];
        FakeBuffer[FakeBufferEnd].dwData = data[i];
        ++FakeBufferEnd;
    }
}


// RemapWrapper: Keyboard remapper
class RemapWrapper : public ProxyInputDevice {
public:
    RemapWrapper(IDirectInputDevice8* device) : ProxyInputDevice(device) {}

    HRESULT _stdcall GetDeviceState(DWORD a, void* b) {
        BYTE bytes[256];
        HRESULT hr = realDevice->GetDeviceState(256, bytes);
        if (hr != DI_OK) {
            return hr;
        }

        BYTE* b2 = (BYTE*)b;
        ZeroMemory(b, 256);
        for (int i = 0; i < 256; i++) {
            if (RemappedKeys[i]) {
                b2[RemappedKeys[i]] |= bytes[i];
            } else {
                b2[i] = bytes[i];
            }
        }
        return DI_OK;
    }

    HRESULT _stdcall GetDeviceData(DWORD a, DIDEVICEOBJECTDATA* b ,DWORD* c, DWORD d) {
        if (*c != 1 || b == NULL) {
            return realDevice->GetDeviceData(a, b, c, d);
        }

        HRESULT hr = realDevice->GetDeviceData(a, b, c, d);
        if (*c != 1 || hr != DI_OK) {
            return hr;
        }

        if (RemappedKeys[b->dwOfs]) {
            b->dwOfs = RemappedKeys[b->dwOfs];
        }

        return hr;
    }
};


// MGEProxyKeyboard: Handles keyboard macros and triggers
class MGEProxyKeyboard : public ProxyInputDevice {
public:
    MGEProxyKeyboard(IDirectInputDevice8* device) : ProxyInputDevice(device) {}

    HRESULT _stdcall GetDeviceState(DWORD a, LPVOID b) {
        // This is a keyboard, so get a list of bytes
        BYTE bytes[MGEINPUT_MAXMACROS];
        HRESULT hr = realDevice->GetDeviceState(256, bytes);
        if (hr != DI_OK) {
            return hr;
        }

        // Copy mouse state to act as an extra 10 keys
        CopyMemory(&bytes[256], &MouseOut, 10);

        // Set modifier bitfield before any macros
        int modifierKeys = 0;
        if (bytes[DIK_LSHIFT] || bytes[DIK_RSHIFT]) {
            modifierKeys |= 1;
        }
        if (bytes[DIK_LCONTROL] || bytes[DIK_RCONTROL]) {
            modifierKeys |= 2;
        }
        if (bytes[DIK_LALT] || bytes[DIK_RALT]) {
            modifierKeys |= 4;
        }
        MGEProxyDirectInput::modifierKeys = modifierKeys;

        // Get any extra key presses
        GlobalHammer = !GlobalHammer;
        const BYTE* hammer = GlobalHammer ? HammerStates : AHammerStates;

        for (DWORD byte = 0; byte < 256; byte++) {
            bytes[byte] |= FakeStates[byte];
            bytes[byte] |= hammer[byte];
            bytes[byte] &= DisallowMask[byte];
            bytes[byte] |= TapStates[byte];
            TapStates[byte] = 0;
        }
        for (DWORD byte = 256; byte < MGEINPUT_MAXMACROS; byte++) {
            bytes[byte] |= FakeStates[byte];
            bytes[byte] |= hammer[byte];
        }

        if (FinishedFake) {
            // Close the console after faking a command (If using console 1 style)
            FinishedFake = false;
            bytes[0x29] = 0x80;
        } else {
            // Process triggers
            DWORD time = GetTickCount();
            for (DWORD trigger = 0; trigger < MGEINPUT_MAXTRIGGERS; trigger++) {
                if (Triggers[trigger].Active && Triggers[trigger].TimeInterval > 0 && TriggerFireTimes[trigger] < time) {
                    for (int i = 0; i < MGEINPUT_MAXMACROS; i++) {
                        bytes[i] |= Triggers[trigger].Data.KeyStates[i];
                    }

                    TriggerFireTimes[trigger] = time + Triggers[trigger].TimeInterval;
                }
            }
            // Process each key for keypresses
            for (DWORD key = 0; key < MGEINPUT_MAXMACROS; key++) {
                if (FakeKeys[key].type != MT_Unused && (bytes[key] & 0x80)) {
                    BYTE last = LastBytes[key] & 0x80;
                    switch (FakeKeys[key].type) {
                    case MT_Console1:
                        if (!last) {
                            bytes[0x29] = 0x80;
                            FakeString(FakeKeys[key].Console.KeyCodes, FakeKeys[key].Console.KeyStates, FakeKeys[key].Console.Length);
                            CloseConsole = true;
                        }
                        break;
                    case MT_Console2:
                        if (!last) {
                            FakeString(FakeKeys[key].Console.KeyCodes, FakeKeys[key].Console.KeyStates, FakeKeys[key].Console.Length);
                            CloseConsole = false;
                        }
                        break;
                    case MT_Hammer1:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte] && GlobalHammer) {
                                bytes[byte] = 0x80;
                            }
                        }
                        break;
                    case MT_Hammer2:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                HammerStates[byte] = 0x80;
                            }
                        }
                        break;
                    case MT_Unhammer:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                HammerStates[byte] = 0x00;
                            }
                        }
                        break;
                    case MT_AHammer1:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte] && !GlobalHammer) {
                                bytes[byte] = 0x80;
                            }
                        }
                        break;
                    case MT_AHammer2:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                AHammerStates[byte] = 0x80;
                            }
                        }
                        break;
                    case MT_AUnhammer:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                AHammerStates[byte] = 0x00;
                            }
                        }
                        break;
                    case MT_Press1:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                bytes[byte] = 0x80;
                            }
                        }
                        break;
                    case MT_Press2:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                FakeStates[byte] = 0x80;
                            }
                        }
                        break;
                    case MT_Unpress:
                        for (DWORD byte = 0; byte < MGEINPUT_MAXMACROS; byte++) {
                            if (FakeKeys[key].Press.KeyStates[byte]) {
                                FakeStates[byte] = 0x00;
                            }
                        }
                        break;
                    case MT_BeginTimer:
                        if (!last) {
                            Triggers[FakeKeys[key].Timer.TimerID].Active = true;
                        }
                        break;
                    case MT_EndTimer:
                        if (!last) {
                            Triggers[FakeKeys[key].Timer.TimerID].Active = false;
                        }
                        break;
                    case MT_Graphics:
                        // Activate on keydown only, except for certain functions which should repeat
                        if ((!last)||(FakeKeys[key].Function.index == GF_IncreaseZoom ||
                                      FakeKeys[key].Function.index == GF_DecreaseZoom ||
                                      FakeKeys[key].Function.index == GF_IncreaseFOV ||
                                      FakeKeys[key].Function.index == GF_DecreaseFOV)) {
                            (FakeFuncs[FakeKeys[key].Function.index])();
                        }
                        break;
                    }
                }
            }
        }
        CopyMemory(b, bytes, a);
        CopyMemory(LastBytes, bytes, MGEINPUT_MAXMACROS);
        CopyMemory(MouseIn, &bytes[256], 10);
        return DI_OK;
    }

    HRESULT _stdcall GetDeviceData(DWORD a, DIDEVICEOBJECTDATA* b, DWORD* c, DWORD d) {
        // This only gets called for keyboards
        if (*c == 1 && FakeBufferEnd > FakeBufferStart) {
            // Inject a fake keypress
            *b = FakeBuffer[FakeBufferStart++];
            if (FakeBufferStart == FakeBufferEnd) {
                if (CloseConsole) {
                    FinishedFake = true;
                    CloseConsole = false;
                }
                FakeBufferStart = 0;
                FakeBufferEnd = 0;
            }
            return DI_OK;
        } else {
            // Read a real keypress
            if (*c > 1 && !CloseConsole) {
                FakeBufferStart = 0;
                FakeBufferEnd = 0;
            }
            return realDevice->GetDeviceData(a, b, c, d);
        }
    }
};


// Mouse read trace (MGE_MOUSE_TRACE=1, or a mgeXE_mouse_trace.txt marker beside Morrowind.exe for a
// hand-launched session): one CSV row per GetDeviceState — QPC ms, lX, lY, result —
// to mgeXE-mouse.csv beside Morrowind.exe. Same clock as the MouseProbe MWSE mod, so each read lines
// up with the frame that consumed it (mgexe-devkit/tools/mouse-probe.py). Answers whether a lumpy
// per-frame count comes from WHEN MW reads or from how the input arrives.
static FILE* mouseTraceFile() {
    static FILE* f = nullptr;
    static bool checked = false;
    if (!checked) {
        checked = true;
        const char* env = std::getenv("MGE_MOUSE_TRACE");
        if ((env && env[0] == '1') || GetFileAttributesA("mgeXE_mouse_trace.txt") != INVALID_FILE_ATTRIBUTES) {
            f = std::fopen("mgeXE-mouse.csv", "w");
            if (f) {
                std::fputs("qpc_ms,lX,lY,hr\n", f);
            }
        }
    }
    return f;
}

static double qpcMs() {
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return double(c.QuadPart) * 1000.0 / double(freq.QuadPart);
}

// ---- Mouse smoothing ---------------------------------------------------------------------------
// WHY: MW turns the view by exactly the raw count DirectInput hands it at its once-per-frame read.
// A 125 Hz mouse sends a report every 8 ms (and skips polls: 15-30 ms gaps), so a ~17 ms frame
// catches one, two or three reports and the per-frame turn swings by +-50% under a steady hand —
// visible as jumps, worst while running (movement advances on MW's steady clock beside it).
// Measured with the MouseProbe mod + rawmouse-probe.ps1: DirectInput's count equals Windows' raw
// stream exactly, frame pacing is even to 2%, MW maps count -> yaw with no scaling. The lumps are the
// report cadence beating against the frame cadence, so only a re-timing can remove them.
//
// WHAT: a 1 kHz poll thread reads the real device and timestamps every report on arrival (QPC). Each
// report's counts accumulated in the sensor since the previous report, so they are spread evenly over
// that span (one nominal report interval when the mouse was idle before). MW's read receives the
// cumulative motion reconstructed at (read time - one report interval): each frame's turn then matches
// the time the hand actually spent moving in it. EXACT conservation: MW receives integer differences of
// the reconstructed total, so over any run it gets every count the mouse sent, no more, no less.
// COST: about one report interval of added delay (8 ms at 125 Hz; ~1-2 ms for a 1000 Hz mouse, where
// the effect is also small). Buttons and the wheel pass through; a click shorter than a frame is held
// until MW has seen it.
#pragma comment(lib, "winmm.lib")
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

class MouseSmoother {
public:
    explicit MouseSmoother(IDirectInputDevice8* dev) : dev(dev) {
        InitializeCriticalSection(&cs);
        const char* d = std::getenv("MGE_MOUSE_SMOOTH_DELAY_MS");   // dev knob: fixed delay instead of auto
        fixedDelayMs = d ? std::atof(d) : 0.0;
    }

    // Every call MW makes on the real device goes through here, so the poll thread never races it.
    struct Lock {
        CRITICAL_SECTION* c;
        explicit Lock(CRITICAL_SECTION* c) : c(c) { EnterCriticalSection(c); }
        ~Lock() { LeaveCriticalSection(c); }
    };
    CRITICAL_SECTION* lock() { return &cs; }

    // Device acquired by MW. NOTHING pending is discarded: counts inside the smoothing window are real
    // motion (at most one delay's worth) and DirectInput itself drops what happens while unacquired.
    // The first build reset the window here, and MW Acquires far more often than it loses the device,
    // so every call threw the window away: "sensitivity is incredibly low".
    void onAcquired() {
        ++acquireCalls;
        if (!acquired && ++acquireTransitions <= 8) {
            LOG::logline("-- Mouse smoothing: acquired (transition %u, %u Acquire calls so far)",
                         acquireTransitions, acquireCalls);
        }
        acquired = true;
        pendingErr = DI_OK;
        if (!thread) {
            thread = CreateThread(NULL, 0, threadMain, this, 0, NULL);
            if (thread) {
                SetThreadPriority(thread, THREAD_PRIORITY_ABOVE_NORMAL);
            }
        }
    }
    void onUnacquired() { acquired = false; }
    void onDeviceGone() { acquired = false; dead = true; }
    bool running() const { return thread != NULL && !dead; }

    // MW's read (caller holds the lock). Returns the error the poll thread saw, if any, so MW's own
    // lost-device handling (re-Acquire) runs as it would without us.
    HRESULT read(DIMOUSESTATE2* out) {
        if (pendingErr != DI_OK) {
            HRESULT hr = pendingErr;
            pendingErr = DI_OK;
            return hr;
        }
        if (!acquired) {
            // Not ours to smooth until MW re-Acquires: the real device answers (and errors) as usual.
            return dev->GetDeviceState(sizeof(DIMOUSESTATE2), out);
        }
        // Pick up anything that arrived since the thread's last poll, with this read's timestamp.
        pollLocked();
        if (pendingErr != DI_OK) {
            HRESULT hr = pendingErr;
            pendingErr = DI_OK;
            return hr;
        }

        double tau = qpcMs() - delayMs();
        double cx, cy;
        evaluate(tau, cx, cy);
        long long ix = (long long)std::floor(cx + 0.5), iy = (long long)std::floor(cy + 0.5);
        out->lX = LONG(ix - emittedX);
        out->lY = LONG(iy - emittedY);
        emittedX = ix;
        emittedY = iy;
        out->lZ = wheel;
        wheel = 0;
        for (int i = 0; i < 8; ++i) {
            out->rgbButtons[i] = latched[i];
            latched[i] = now[i];
        }
        // Drop reports the reconstruction has fully passed.
        int drop = 0;
        while (drop < count && at(drop).t <= tau) {
            ++drop;
        }
        head = (head + drop) % kRing;
        count -= drop;
        if (++reads % 3000 == 0 && reads <= 30000) {
            LOG::logline("-- Mouse smoothing: %u reads, %u Acquire calls, report interval %.2f ms, delay %.2f ms, "
                         "pending %lld,%lld counts", reads, acquireCalls, interval, delayMs(),
                         cumX - emittedX, cumY - emittedY);
        }
        return DI_OK;
    }

    double intervalMs() const { return interval; }

private:
    struct Report {
        double t, s;              // arrival, start of the span its counts accumulated in
        long long px, py, cx, cy; // cumulative counts before / after it
    };
    static const int kRing = 1024;   // 8 s of a 125 Hz mouse between two MW reads (a load screen)
    static const int kGaps = 32;

    IDirectInputDevice8* dev;
    CRITICAL_SECTION cs;
    HANDLE thread = NULL;
    volatile bool dead = false;
    bool acquired = false;
    HRESULT pendingErr = DI_OK;
    Report ring[kRing];
    int head = 0, count = 0;
    long long cumX = 0, cumY = 0, emittedX = 0, emittedY = 0;
    LONG wheel = 0;
    BYTE now[8] = {}, latched[8] = {};
    double lastT = -1e30;
    double gaps[kGaps] = {};
    int gapN = 0, gapHead = 0;
    double interval = 8.0;           // median report gap; 125 Hz until measured
    double fixedDelayMs = 0.0;
    unsigned acquireCalls = 0, acquireTransitions = 0, reads = 0;

    Report& at(int i) { return ring[(head + i) % kRing]; }

    double delayMs() const {
        // One report interval plus a millisecond of arrival jitter: the newest report then almost
        // always lies past the reconstruction point, so a frame is not starved waiting for it.
        return fixedDelayMs > 0.0 ? fixedDelayMs : interval + 1.0;
    }

    void addReport(double t, LONG dx, LONG dy) {
        double gap = t - lastT;
        if (gap > 0.0 && gap < 50.0) {
            gaps[gapHead] = gap;
            gapHead = (gapHead + 1) % kGaps;
            if (gapN < kGaps) {
                ++gapN;
            }
            double s[kGaps];
            std::memcpy(s, gaps, sizeof(double) * gapN);
            std::nth_element(s, s + gapN / 2, s + gapN);
            interval = (std::min)(20.0, (std::max)(0.5, s[gapN / 2]));
        }
        // A skipped poll (a gap of a few intervals) still carries the counts of the whole gap; a report
        // after the mouse sat still carries one interval's worth.
        double span = (gap > 0.0 && gap <= 4.0 * interval) ? gap : interval;

        if (count == kRing) {   // MW stopped reading: forget the oldest, its counts stay in the totals
            head = (head + 1) % kRing;
            --count;
        }
        Report& r = ring[(head + count) % kRing];
        r.t = t;
        r.s = t - span;
        r.px = cumX;
        r.py = cumY;
        cumX += dx;
        cumY += dy;
        r.cx = cumX;
        r.cy = cumY;
        ++count;
        lastT = t;
    }

    // Cumulative motion at time tau: reports are ordered and their spans do not overlap.
    void evaluate(double tau, double& x, double& y) {
        if (count == 0) {
            x = double(cumX);
            y = double(cumY);
            return;
        }
        x = double(at(0).px);
        y = double(at(0).py);
        for (int i = 0; i < count; ++i) {
            const Report& r = at(i);
            if (tau >= r.t) {
                x = double(r.cx);
                y = double(r.cy);
                continue;
            }
            if (tau > r.s) {
                double f = (tau - r.s) / (r.t - r.s);
                x = double(r.px) + double(r.cx - r.px) * f;
                y = double(r.py) + double(r.cy - r.py) * f;
            }
            break;
        }
    }

    void pollLocked() {
        if (!acquired || dead || pendingErr != DI_OK) {
            return;
        }
        DIMOUSESTATE2 st;
        HRESULT hr = dev->GetDeviceState(sizeof(DIMOUSESTATE2), &st);
        if (hr != DI_OK) {
            pendingErr = hr;   // handed to MW's next read, which re-Acquires
            acquired = false;
            return;
        }
        if (st.lX || st.lY) {
            addReport(qpcMs(), st.lX, st.lY);
        }
        wheel += st.lZ;
        for (int i = 0; i < 8; ++i) {
            now[i] = st.rgbButtons[i];
            latched[i] |= st.rgbButtons[i];
        }
    }

    static DWORD WINAPI threadMain(void* p) {
        MouseSmoother* self = (MouseSmoother*)p;
        HANDLE timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        bool coarse = false;
        if (!timer) {
            // Pre-1803 Windows: a 1 ms Sleep needs the 1 ms system timer.
            coarse = true;
            timeBeginPeriod(1);
        }
        LOG::logline("-- Mouse smoothing: poll thread up (%s timer)", coarse ? "timeBeginPeriod" : "high-resolution");
        while (!self->dead) {
            if (timer) {
                LARGE_INTEGER due;
                due.QuadPart = -10000;   // 1 ms, relative
                SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
                WaitForSingleObject(timer, 100);
            } else {
                Sleep(1);
            }
            Lock l(&self->cs);
            self->pollLocked();
        }
        if (timer) {
            CloseHandle(timer);
        } else {
            timeEndPeriod(1);
        }
        return 0;
    }
};

// MGEProxyMouse: Maps mouse buttons to macro trigger inputs
class MGEProxyMouse : public ProxyInputDevice {
public:
    DWORD deviceType;
    MouseSmoother* smoother = nullptr;

    MGEProxyMouse(IDirectInputDevice8* device) : ProxyInputDevice(device) {
        if (Configuration.Input.MouseSmoothing) {
            smoother = new MouseSmoother(device);
        }
        LOG::logline("-- Mouse smoothing %s", smoother ? "on" : "off");
    }

    HRESULT _stdcall SetCooperativeLevel(HWND a, DWORD b) {
        LOG::logline("-- Mouse cooperative level 0x%x (%s%s%s)", b,
                     (b & DISCL_EXCLUSIVE) ? "exclusive " : "", (b & DISCL_NONEXCLUSIVE) ? "nonexclusive " : "",
                     (b & DISCL_FOREGROUND) ? "foreground" : "background");
        if (!smoother) {
            return realDevice->SetCooperativeLevel(a, b);
        }
        MouseSmoother::Lock l(smoother->lock());
        return realDevice->SetCooperativeLevel(a, b);
    }

    HRESULT _stdcall Acquire() {
        if (!smoother) {
            return realDevice->Acquire();
        }
        MouseSmoother::Lock l(smoother->lock());
        HRESULT hr = realDevice->Acquire();
        if (SUCCEEDED(hr)) {
            smoother->onAcquired();
        }
        return hr;
    }

    HRESULT _stdcall Unacquire() {
        if (!smoother) {
            return realDevice->Unacquire();
        }
        MouseSmoother::Lock l(smoother->lock());
        smoother->onUnacquired();
        return realDevice->Unacquire();
    }

    ULONG _stdcall Release() {
        if (!smoother) {
            return realDevice->Release();
        }
        MouseSmoother::Lock l(smoother->lock());
        ULONG r = realDevice->Release();
        if (r == 0) {
            smoother->onDeviceGone();   // the poll thread sees it under the lock and exits
        }
        return r;
    }

    HRESULT _stdcall SetDataFormat(const DIDATAFORMAT* a) {
        if (!smoother) {
            return realDevice->SetDataFormat(a);
        }
        MouseSmoother::Lock l(smoother->lock());
        return realDevice->SetDataFormat(a);
    }

    HRESULT _stdcall SetProperty(REFGUID a, const DIPROPHEADER* b) {
        if (!smoother) {
            return realDevice->SetProperty(a, b);
        }
        MouseSmoother::Lock l(smoother->lock());
        return realDevice->SetProperty(a, b);
    }

    HRESULT _stdcall GetDeviceState(DWORD a, LPVOID b) {
        DIMOUSESTATE2* mouseState = (DIMOUSESTATE2*)b;
        HRESULT hr;
        if (smoother && smoother->running()) {
            MouseSmoother::Lock l(smoother->lock());
            hr = smoother->read(mouseState);
        } else if (smoother) {
            MouseSmoother::Lock l(smoother->lock());
            hr = realDevice->GetDeviceState(sizeof(DIMOUSESTATE2), mouseState);
        } else {
            hr = realDevice->GetDeviceState(sizeof(DIMOUSESTATE2), mouseState);
        }
        if (FILE* f = mouseTraceFile()) {
            std::fprintf(f, "%.3f,%ld,%ld,0x%lx\n", qpcMs(), hr == DI_OK ? mouseState->lX : 0L,
                         hr == DI_OK ? mouseState->lY : 0L, (unsigned long)hr);
            static unsigned n = 0;
            if (++n % 256 == 0) {
                std::fflush(f);
            }
        }
        if (hr != DI_OK) {
            return hr;
        }

        // Notify application of clicks
        MGEProxyDirectInput::mouseClick = MouseOut[0] & ~mouseState->rgbButtons[0];

        // Map mousewheel to macro triggers 8/9
        if (mouseState->lZ>0) {
            MouseOut[8] = 0x80;
            MouseOut[9] = 0;
        } else if (mouseState->lZ<0) {
            MouseOut[8] = 0;
            MouseOut[9] = 0x80;
        } else {
            MouseOut[8] = 0;
            MouseOut[9] = 0;
        }

        for (DWORD i = 0; i < 8; i++) {
            MouseOut[i] = mouseState->rgbButtons[i];
            mouseState->rgbButtons[i] |= MouseIn[i];
            mouseState->rgbButtons[i] &= DisallowMask[i+256];
            mouseState->rgbButtons[i] |= TapStates[i+256];
            TapStates[i+256] = 0;
        }

        return DI_OK;
    }
};


// MGEProxyKeyboardAltCombat: Keyboard component of Daggerfall-like combat input
class MGEProxyKeyboardAltCombat : public MGEProxyKeyboard {
public:
    MGEProxyKeyboardAltCombat(IDirectInputDevice8* device) : MGEProxyKeyboard(device) {}

    HRESULT _stdcall GetDeviceState(DWORD a, void* b) {
        HRESULT hr = MGEProxyKeyboard::GetDeviceState(a, b);
        if (hr != DI_OK) {
            return hr;
        }

        // Don't run combat input mode when a menu is up
        auto mwBridge = MWBridge::get();
        if (!mwBridge->IsLoaded() || mwBridge->IsMenu()) {
            return DI_OK;
        }

        // We only want to modify keyboard input when the player has the mouse held down
        if (AltCombat.attackType && AltCombat.attackType != State_NOMOTION) {
            BYTE* keyState = (BYTE*)b;

            // Read scancodes for movement keybinds (which can change during play)
            int forward = mwBridge->getKeybindCode(0);
            int back = mwBridge->getKeybindCode(1);
            int left = mwBridge->getKeybindCode(2);
            int right = mwBridge->getKeybindCode(3);

            // Set all movement keys to up state
            keyState[forward] = keyState[back] = keyState[left] = keyState[right] = 0;

            // Then set appropriate keys to pressed depending on what type of attack is being made
            if (GlobalHammer) {
                // AltCombat.attackType == State_CHOP -> no key required
                if (AltCombat.attackType == State_SLASH) {
                    keyState[left] = 0x80;
                }
                if (AltCombat.attackType == State_PIERCE) {
                    keyState[forward] = 0x80;
                }
            } else {
                // AltCombat.attackType == State_CHOP -> no key required
                if (AltCombat.attackType == State_SLASH) {
                    keyState[right] = 0x80;
                }
                if (AltCombat.attackType == State_PIERCE) {
                    keyState[back] = 0x80;
                }
            }

            // Tell the mouse proxy that a swing is ready, so to intiate attack
            AltCombat.directionPressed = true;
        }
        return DI_OK;
    }
};


// MGEProxyMouseAltCombat: Mouse component of Daggerfall-like combat input
class MGEProxyMouseAltCombat : public MGEProxyMouse {
public:
    MGEProxyMouseAltCombat(IDirectInputDevice8* device) : MGEProxyMouse(device) {}

    HRESULT _stdcall GetDeviceState(DWORD a, void* b) {
        HRESULT hr = MGEProxyMouse::GetDeviceState(a, b);
        if (hr != DI_OK) {
            return hr;
        }

        // Don't run combat input mode when a menu is up
        auto mwBridge = MWBridge::get();
        if (!mwBridge->IsLoaded() || mwBridge->IsMenu()) {
            return DI_OK;
        }

        // Capture mouse movement while mouse is pressed
        // Skip/cancel if a ranged weapon is equipped
        DIMOUSESTATE2* mouseState = (DIMOUSESTATE2*)b;
        bool ranged = mwBridge->getPlayerWeapon() >= 9;

        if (mouseState->rgbButtons[0] && !ranged) {
            // If the difference between x and y movement is greater than maxGap, prefer slash over chop
            if (abs(mouseState->lX) > abs(mouseState->lY)+maxGap) {
                mouseState->lY = 0;
            }

            bool slash = abs(mouseState->lX) > altSensitivity;
            bool pierce = abs(mouseState->lY) > altSensitivity;

            int attack = 0;   // Which direction has the mouse moved
            if (mouseState->lX > altSensitivity) {
                attack |= 0x0001;
            }
            if (mouseState->lX < -altSensitivity) {
                attack |= 0x0010;
            }
            if (mouseState->lY > altSensitivity) {
                attack |= 0x0100;
            }
            if (mouseState->lY < -altSensitivity) {
                attack |= 0x1000;
            }

            if (AltCombat.directionPressedLast && attack == AltCombat.lastAttack && attack != 0) {
                AltCombat.directionPressed = true;
            }

            if (attack == AltCombat.lastAttack || attack == 0) {
                AltCombat.attackType = State_NOMOTION;  // Can't attack by moving the mouse in the same direction twice
            } else {
                // Set attack type appropriately depending on mouse movement
                if (slash && pierce) {
                    AltCombat.attackType = State_CHOP;
                } else if (slash) {
                    AltCombat.attackType = State_SLASH;
                } else if (pierce) {
                    AltCombat.attackType = State_PIERCE;
                } else {
                    // This differentiates between not having the mouse button down, and having the mouse down but not moving it
                    AltCombat.attackType = State_NOMOTION;
                }
                AltCombat.lastAttack = attack;
            }

            // Don't pass mouse movement and left button state to Morrowind
            mouseState->lX = 0;
            mouseState->lY = 0;
            mouseState->rgbButtons[0] = 0;

            // If the correct movement key is down then press the left mouse button
            if (AltCombat.directionPressed) {
                mouseState->rgbButtons[0] = 0x80;
            }

            AltCombat.directionPressedLast = AltCombat.directionPressed;
        } else {
            // Mouseup state passes through to finish attacks

            // Reset alt combat on mouse up / ranged
            AltCombat.directionPressed = false;
            AltCombat.directionPressedLast = false;
            AltCombat.attackType = State_NONE;
            AltCombat.lastAttack = 0;
        }
        return DI_OK;
    }
};



IDirectInputDevice8* MGEProxyDirectInput::factoryProxyInput(IDirectInputDevice8* device, REFGUID g) {
    if (g == GUID_SysKeyboard) {
        if (Configuration.Input.AltCombat) {
            device = new MGEProxyKeyboardAltCombat(device);
        } else {
            device = new MGEProxyKeyboard(device);
        }

        if (Configuration.Input.Remap[0] != 0) {
            device = new RemapWrapper(device);
        }
    } else if (g == GUID_SysMouse) {
        if (Configuration.Input.AltCombat) {
            device = new MGEProxyMouseAltCombat(device);
        } else {
            device = new MGEProxyMouse(device);
        }
    }

    return device;
}


// Input config parser

static bool entryParse(char* text, char prefix, size_t* key, const char** values, size_t* value_count) {
    // Split line into <key>=<values>
    char* sep = std::strchr(text, '=');
    if (sep == NULL || *text != prefix) {
        return false;
    }

    // In-place convert comma-separated list into null-terminated strings
    size_t c = 0;
    for (char* v = sep + 1; v; ++c) {
        v = std::strchr(v, ',');
        if (v) {
            *v++ = 0;
        }
    }

    *key = std::atoi(text + 1);
    *values = sep + 1;
    *value_count = c;
    return true;
}


static const char* entryNextValue(const char* s) {
    return s + strlen(s) + 1;
}

static void loadInputSettings() {
    size_t seek;
    for (char* line = Configuration.Input.Macros; *line; line += seek) {
        size_t key, value_count;
        const char* values;
        seek = strlen(line) + 1;

        if (!entryParse(line, 'M', &key, &values, &value_count)) {
            continue;
        }
        if (value_count < 2 || key >= MGEINPUT_MAXMACROS) {
            continue;
        }

        sFakeKey* macro = &FakeKeys[key];
        for (const MacroTypeLabel* x = macroTypeLabels; x->label; ++x) {
            if (std::strcmp(values, x->label) == 0) {
                macro->type = x->type;
                break;
            }
        }

        switch (macro->type) {
        case MT_Console1:
        case MT_Console2:
            macro->Console.Length = 0;
            for (size_t i = 2; i < value_count; i += 2, ++macro->Console.Length) {
                values = entryNextValue(values);
                macro->Console.KeyCodes[macro->Console.Length] = std::atoi(values);
                values = entryNextValue(values);
                macro->Console.KeyStates[macro->Console.Length] = (std::strcmp(values, "True") == 0) ? 0x80 : 0;
            }
            break;
        case MT_Press1:
        case MT_Press2:
        case MT_Unpress:
        case MT_Hammer1:
        case MT_Hammer2:
        case MT_Unhammer:
        case MT_AHammer1:
        case MT_AHammer2:
        case MT_AUnhammer:
            for (size_t i = 1; i < value_count; ++i) {
                values = entryNextValue(values);
                macro->Press.KeyStates[std::atoi(values)] = 0x80;
            }
            break;
        case MT_BeginTimer:
        case MT_EndTimer:
            macro->Timer.TimerID = std::atoi(entryNextValue(values));
            break;
        case MT_Graphics:
            macro->Function.index = std::atoi(entryNextValue(values));
            break;
        }
    }

    for (char* line = Configuration.Input.Triggers; *line; line += seek) {
        size_t key, value_count;
        const char* values;
        seek = strlen(line) + 1;

        if (!entryParse(line, 'T', &key, &values, &value_count)) {
            continue;
        }
        if (value_count < 2 || key >= MGEINPUT_MAXTRIGGERS) {
            continue;
        }

        sFakeTrigger* trigger = &Triggers[key];

        trigger->Active = (std::strcmp(values, "True") == 0);
        values = entryNextValue(values);
        trigger->TimeInterval = 1000 * std::atoi(values);

        for (size_t i = 2; i < value_count; ++i) {
            values = entryNextValue(values);
            trigger->Data.KeyStates[std::atoi(values)] = 0x80;
        }
    }

    for (char* line = Configuration.Input.Remap; *line; line += seek) {
        size_t key, value_count;
        const char* values;
        seek = strlen(line) + 1;

        if (!entryParse(line, 'R', &key, &values, &value_count)) {
            continue;
        }
        if (value_count != 1 || key >= 256) {
            continue;
        }

        RemappedKeys[key] = std::atoi(values);
    }
}
