
#include "statusoverlay.h"
#include "configuration.h"
#include "mged3d8device.h"
#include "proxydx/d3d9header.h"

#include <cstdio>



namespace StatusOverlay {
    char statusText[512];
    char fpsText[16];
    char frameNumText[32];
    DWORD statusTimeout;
    int currentPriority;
    D3DCOLOR statusColour;

    ID3DXFont* font;
    ID3DXSprite* sprite;
    RECT statusRect, shadowRect, fpsRect, frameNumRect;
}

const D3DCOLOR colWhite = 0xffffffff, colRed = 0xffff2222, colShadow = 0xc0000000;
const char* fontFace = "Lucida Console";
const int fontHeight = 14;

bool StatusOverlay::init(IDirect3DDevice9* device) {
    D3DXCreateFont(device, fontHeight, 0, 400, 1, false, ANSI_CHARSET, OUT_DEFAULT_PRECIS,
                   DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, fontFace, &font);
    D3DXCreateSprite(device, &sprite);

    statusRect = {8, 10, 635, 25};
    shadowRect = {statusRect.left+1, statusRect.top+1, statusRect.right+1, statusRect.bottom+1};
    fpsRect = {8, 35, 160, 50};
    frameNumRect = {60, 35, 200, 50};  // Right of FPS
    return true;
}

void StatusOverlay::release() {
    sprite->Release();
    font->Release();
}

void StatusOverlay::show(IDirect3DDevice9* device) {
    if (!(Configuration.MGEFlags & DISPLAY_MESSAGES)) {
        return;
    }

    // Frame number always visible for debugging; FPS and status depend on config
    bool showAnything = frameNumText[0] || (Configuration.MGEFlags & FPS_COUNTER) || statusTimeout;
    if (showAnything) {
        sprite->Begin(D3DXSPRITE_ALPHABLEND);

        // Always show frame number when populated
        if (frameNumText[0]) {
            font->DrawTextA(sprite, frameNumText, -1, &frameNumRect, DT_NOCLIP, colWhite);
        }

        if (Configuration.MGEFlags & FPS_COUNTER) {
            font->DrawTextA(sprite, fpsText, -1, &fpsRect, DT_NOCLIP, colWhite);
        }

        if (statusText[0] != 0) {
            if (GetTickCount() < statusTimeout) {
                font->DrawTextA(sprite, statusText, -1, &shadowRect, DT_NOCLIP, colShadow);
                font->DrawTextA(sprite, statusText, -1, &statusRect, DT_NOCLIP, statusColour);
            } else {
                statusTimeout = 0;
                currentPriority = 0;
            }
        }
        sprite->End();
    }
}

void StatusOverlay::setStatus(const char* s, int priority) {
    if (priority >= currentPriority) {
        std::snprintf(statusText, sizeof(statusText), "%s", s);
        statusTimeout = GetTickCount() + priority * Configuration.StatusTimeout;
        currentPriority = priority;
        statusColour = currentPriority >= Priority::PriorityWarning ? colRed : colWhite;
    }
}

void StatusOverlay::setFPS(float fps) {
    std::snprintf(fpsText, sizeof(fpsText), "%4.0f", fps);
}

void StatusOverlay::setFrameNumber(int frameNum) {
    std::snprintf(frameNumText, sizeof(frameNumText), "F:%d", frameNum);
}

void StatusOverlay::showLastStatus() {
    statusTimeout = GetTickCount() + Configuration.StatusTimeout;
}
