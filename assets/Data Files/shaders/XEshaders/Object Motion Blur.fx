// Object Motion Blur with Silhouette Expansion and Falloff
// Based on McGuire reconstruction filter

#define MAX_SAMPLES 4
#define MIN_SAMPLES 2  

// ** ADJUSTABLE VARIABLES
static float blur_scale = 0.1;      // Multiplier for blur intensity
static float max_blur = 0.05;      // Maximum blur soft limit
static float soft_z_extent = 0.05;  // Depth comparison softness
static float expansion_radius = 24.0; // Radius to search for moving objects
// **

float2 rcpres;
float time;

texture lastshader;
texture lastpass;
texture depthframe;
texture velocityframe;

sampler sDepth = sampler_state { texture = <depthframe>; addressu = clamp; addressv = clamp; magfilter = point; minfilter = point; };
sampler sVelocity = sampler_state { texture = <velocityframe>; addressu = clamp; addressv = clamp; magfilter = linear; minfilter = linear; };
sampler sFrame = sampler_state { texture = <lastshader>; magfilter = point; minfilter = point; };
sampler sPass = sampler_state { texture = <lastpass>; magfilter = linear; minfilter = linear; addressu = clamp; addressv = clamp; };

float2 decodeVelocity(float2 encoded) {
    encoded /= 50.0;
    float2 sign_v = sign(encoded);
    return sign_v * pow(abs(encoded), 1.0 / 3.0);
}

float hash(float2 p) {
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

// Multi-step search with distance falloff
float2 getDilatedVelocity(float2 tex, float centerDepth) {
    float2 centerVel = decodeVelocity(tex2D(sVelocity, tex).rg);
    float maxLen = length(centerVel);
    float2 maxVel = centerVel;

    // 4 directions (Cross pattern)
    float2 searchDirs[4] = {
        float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1)
    };

    // We step outward in 3 increments to create a smooth gradient
    [unroll]
    for (int i = 0; i < 4; i++) {
        [unroll]
        for (float step = 0.33; step <= 1.0; step += 0.33) {
            float2 offset = searchDirs[i] * (expansion_radius * step) * rcpres;
            float2 nTex = tex + offset;
            
            float2 nVel = decodeVelocity(tex2D(sVelocity, nTex).rg);
            float nDepth = tex2D(sDepth, nTex).r;
            float nLen = length(nVel);
            
            // Only borrow velocity if the neighbor is strictly in the foreground
            if (nDepth < centerDepth - 0.005) {
                // Linear falloff: weight goes from 1.0 (close) to 0.0 (at expansion radius)
                float weight = 1.0 - step; 
                float candidateLen = nLen * weight;
                
                // If this scaled velocity is stronger than what we have, use it
                if (candidateLen > maxLen) {
                    maxLen = candidateLen;
                    maxVel = nVel * weight; // Scale the velocity length down
                }
            }
        }
    }

    return maxVel;
}

float4 Mask(in float2 tex : TEXCOORD) : COLOR0
{
    return tex2D(sFrame, tex);
}

float4 ObjectMotionBlur(in float2 tex : TEXCOORD) : COLOR0
{
    float centerDepth = tex2D(sDepth, tex).r;

    // Get velocity scaled by distance from silhouettes
    float2 velocity = getDilatedVelocity(tex, centerDepth);
    velocity *= blur_scale;

    float velLen = length(velocity);
    
    // Early out for static pixels
    if (velLen < 0.0005) return tex2D(sPass, tex);

    // Soft clamp for max blur
    // float softMax = max_blur;
    // if (velLen > softMax) {
        // velocity = (velocity / velLen) * softMax;
        // velLen = softMax;
    // }

    // Adaptive Samples (Shading matches velocity scale)
    float velPixels = velLen / rcpres.x;
    int nSamples = (int)clamp(velPixels * 0.5, (float)MIN_SAMPLES, (float)MAX_SAMPLES);

    float jitter = hash(tex / rcpres + time) - 0.5;
    float4 color = 0;
    float totalWeight = 0;

    for (int i = -MAX_SAMPLES; i <= MAX_SAMPLES; i++)
    {
        float t = (float(i) + jitter) / float(MAX_SAMPLES);
        float2 offset = velocity * t;
        float2 samplePos = tex + offset;
        
        float4 sampleColor = tex2D(sPass, samplePos);
        float sampleDepth = tex2D(sDepth, samplePos).r;

        // McGuire Reconstruction Weights
        float fWeight = saturate(1.0 - (sampleDepth - centerDepth) / soft_z_extent);
        float bWeight = saturate(1.0 - (centerDepth - sampleDepth) / soft_z_extent);
        
        float weight = max(fWeight, bWeight);
        
        // Gaussian blur falloff
        weight *= exp(-3.0 * t * t);

        color += sampleColor * weight;
        totalWeight += weight;
    }

    return color / max(totalWeight, 0.0001);
}

technique T0 < string MGEinterface = "MGE XE 0"; >
{
    pass { PixelShader = compile ps_3_0 Mask(); }
    pass { PixelShader = compile ps_3_0 ObjectMotionBlur(); }
}