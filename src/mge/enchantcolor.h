#pragma once

// MW's per-item enchanted-glow COLOUR.
//
// The caustic environment map MW lays over an enchanted item is greyscale (verified: the 32
// magicitem\caust*.dds frames average RGB 19,19,19); the colour comes from the item's enchantment,
// and MW feeds it to the glow draw as the D3D material diffuse. Nothing in the scene graph carries
// it — a live probe over 25 enchanted shapes found dif=(1,1,1), emis=(0,0,0) and the artist's own
// ambient on every one — which is exactly why the DX9 path gets the tint for free (its FFE reads
// device state) and the Forge path had to reconstruct it.
//
// The reconstruction is the same one MW performs and OpenMW reimplements as getEnchantmentColor:
//
//     scene node -> TES3 object -> Enchantment -> effects[0].effectID -> MGEF lighting RGB
//
// Every link is cheap and, crucially, the FIRST one is a direct read: each item MW attaches the
// glow to turns out to own its own TES3 reference (measured — 'CLONE icicle' -> WEAP 'icicle',
// 'CLONE thief_ring' -> CLOT 'thief_ring', ...), so no engine detour is needed to find the item.
//
// The MGEF table is read from the PLUGIN FILES rather than from the running engine on purpose:
// TES3::NonDynamicData::magicEffects (0x5C8) is an inline array in vanilla but a
// MagicEffectController* once MWSE installs its custom controller (MWSE_CUSTOM_EFFECTS is true),
// and both are pointers at the same offset with no reliable way to tell them apart. Reading the
// files is deterministic, works with or without MWSE, and honours mod overrides for free.

#include <cstdint>

namespace MGE::EnchantColor {

    // Parse MGEF records from every plugin in Morrowind.ini's [Game Files], in load order (later
    // plugins override earlier ones — the same precedence the engine applies). Idempotent and
    // lazy: the first colorForObject() call triggers it. Safe to call early; returns false if the
    // ini or the Data Files directory could not be read, in which case every lookup falls back to
    // the caller's default colour and the glow simply stays untinted.
    bool init();

    // The glow colour for a TES3 object (the base object behind a reference), as linear-ish 0..1
    // RGB straight from the MGEF record. Returns false and leaves `outRGB` untouched when the
    // object is null, is not enchantable, carries no enchantment, or its first effect has no MGEF
    // entry — so a caller can keep its own default without a second branch.
    //
    // `tes3Object` is void* because MGE consumes only SharedSE, where the TES3 object types are
    // opaque; the two reads this needs (the vtable's getEnchantment at 0xD0 and Enchantment's
    // effects[] at 0x34) are at MWSE-documented offsets, exactly like referenceLiveKind's.
    bool colorForObject(const void* tes3Object, float outRGB[3]);

    // How many MGEF colours were loaded (diagnostic; 0 means init() failed or found nothing).
    unsigned effectCount();

}
