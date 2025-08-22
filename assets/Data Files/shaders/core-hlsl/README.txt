MGE XE HLSL Pipeline Implementation
===================================

This directory contains the new HLSL shader implementation for MGE XE, designed to replace the legacy ID3DXEffect system.

Current Status:
- ✅ Dual-path architecture implemented (old and new systems coexist)
- ✅ Updated skin() function with float-based blend state (your requested change)
- ✅ HLSL compilation pipeline added to ffeshader.cpp
- ✅ Runtime switching via Configuration.UseHLSLPipeline flag
- ✅ Basic shader variants for rigid and skinned objects

Key Changes:
1. skin() function now uses float4 vertexBlendState instead of int
2. Blend weight access uses .xyzw swizzling instead of array indexing
3. D3DCompile replaces D3DXCreateEffectFromFile for shader compilation
4. Separate vertex/pixel shader entry points instead of technique blocks

Files:
- XE Common_HLSL.fx: Shared functions and structures (includes updated skin() function)
- XE FixedFuncEmu_HLSL.fx: Fixed function emulation shaders
- README.txt: This file

Usage:
Set Configuration.UseHLSLPipeline = true to enable the new pipeline.
The system will automatically fall back to the old pipeline if set to false.

Testing:
Both pipelines should render identically. Use this for comparison testing
during the migration process.

Next Steps:
1. Add full constant buffer setup in renderMorrowindHLSL()
2. Convert remaining core shader files (XE Main, XE Depth, etc.)
3. Test with different shader variants and edge cases
4. Performance validation
5. Remove old D3DXEffect code once migration is complete