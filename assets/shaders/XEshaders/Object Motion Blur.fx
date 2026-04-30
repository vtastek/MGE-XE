// Object Motion Blur
// AAA-quality implementation based on McGuire reconstruction filter
// Techniques from NFS 2015, CoD:AW, John Chapman

// **
// ** ADJUSTABLE VARIABLES

#define MAX_SAMPLES 16 // Maximum samples for quality
#define MIN_SAMPLES 3  // Minimum samples

static float blur_scale = 1.0; // Multiplier for blur intensity
static float mask_distance = 33.0; // Ignore near objects (hands/weapon)
static float max_blur = 0.08; // Maximum blur in NDC units
static float velocity_threshold = 0.0005; // Minimum velocity to apply blur
static float soft_z_extent = 0.1; // Depth comparison softness

// ** END OF
// **

float2 rcpres;

texture lastshader;
texture lastpass;
texture depthframe;
texture velocityframe;

sampler sDepth = sampler_state { texture = <depthframe>; addressu = clamp; addressv = clamp; magfilter = point; minfilter = point; };
sampler sVelocity = sampler_state { texture = <velocityframe>; addressu = clamp; addressv = clamp; magfilter = point; minfilter = point; };
sampler sFrame = sampler_state { texture = <lastshader>; magfilter = point; minfilter = point; };
sampler sPass = sampler_state { texture = <lastpass>; magfilter = linear; minfilter = linear; addressu = clamp; addressv = clamp; };

// Decode velocity with precision redistribution (inverse of pow3 encoding)
// XE Depth.fx encodes as: sign(v) * pow(abs(v), 3.0) * 50.0
// We decode as: sign(v) * pow(abs(v / 50.0), 1/3.0)
float2 decodeVelocity(float2 encoded) {
    encoded /= 50.0;
    float2 sign_v = sign(encoded);
    return sign_v * pow(abs(encoded), 1.0 / 3.0);
}

// Soft depth comparison - returns [0,1] weight based on depth difference
// Foreground objects can blur over background, not vice-versa
float softDepthCompare(float za, float zb) {
    return saturate(1.0 - (za - zb) / soft_z_extent);
}

// Cone weight - samples at center contribute more than edges
float cone(float dist, float velLen) {
    return saturate(1.0 - dist / velLen);
}

// Cylinder weight with soft depth
float cylinder(float dist, float velLen) {
    return 1.0 - smoothstep(0.95 * velLen, 1.05 * velLen, dist);
}

// Get dominant velocity from 3x3 neighborhood (simplified TileMax)
// Returns decoded velocity (already processed through pow 1/3)
float2 getNeighborMaxVelocity(float2 tex) {
    float2 maxEncoded = tex2D(sVelocity, tex).rg;
    float maxLen = dot(maxEncoded, maxEncoded);

    // Sample 4 neighbors (cross pattern for performance)
    float2 offsets[4] = {
        float2(-2.0, 0.0) * rcpres,
        float2(2.0, 0.0) * rcpres,
        float2(0.0, -2.0) * rcpres,
        float2(0.0, 2.0) * rcpres
    };

    for (int i = 0; i < 4; i++) {
        float2 vel = tex2D(sVelocity, tex + offsets[i]).rg;
        float len = dot(vel, vel);
        if (len > maxLen) {
            maxEncoded = vel;
            maxLen = len;
        }
    }

    return decodeVelocity(maxEncoded);
}

float4 Mask(in float2 tex : TEXCOORD) : COLOR0
{
    float depth = tex2D(sDepth, tex).r;
    float mask = (depth > mask_distance);
    return mask ? tex2D(sFrame, tex) : 0;
}

float4 ObjectMotionBlur(in float2 tex : TEXCOORD) : COLOR0
{
    float centerDepth = tex2D(sDepth, tex).r;

    // Skip near objects (hands/weapon)
    if (centerDepth < mask_distance) {
        return tex2D(sFrame, tex);
    }

    // Get velocity - check neighborhood for edge blur
    float2 centerVel = decodeVelocity(tex2D(sVelocity, tex).rg);
    float2 neighborVel = getNeighborMaxVelocity(tex); // Already decoded

    // Use whichever velocity is larger (allows blur to extend past silhouette)
    float2 velocity = (dot(neighborVel, neighborVel) > dot(centerVel, centerVel))
                      ? neighborVel : centerVel;

    // Apply blur scale
    velocity *= blur_scale;

    // Calculate velocity magnitude
    float velLen = length(velocity);

    // Skip if velocity too small
    if (velLen < velocity_threshold) {
        return tex2D(sFrame, tex);
    }

    // Clamp to max blur
    if (velLen > max_blur) {
        velocity = velocity / velLen * max_blur;
        velLen = max_blur;
    }

    // Adaptive sample count based on velocity (more samples for faster motion)
    float velPixels = velLen / rcpres.x; // Convert to pixel distance
    int nSamples = clamp(int(velPixels * 0.5), MIN_SAMPLES, MAX_SAMPLES);

    // Sample along velocity direction, centered on current pixel
    float4 color = tex2D(sPass, tex);
    float totalWeight = 1.0; // Center sample always has weight 1

    float2 stepVec = velocity / float(nSamples);

    for (int i = 1; i <= nSamples; i++)
    {
        // Sample in both directions from center
        float t = float(i) / float(nSamples);

        float2 offset = stepVec * float(i);

        // Forward sample
        float2 samplePos = tex + offset;
        float sampleDepth = tex2D(sDepth, samplePos).r;

        // Weight based on depth relationship and distance from center
        float depthWeight = softDepthCompare(centerDepth, sampleDepth);
        float coneWeight = cone(length(offset), velLen);
        float weight = depthWeight * coneWeight;

        color += tex2D(sPass, samplePos) * weight;
        totalWeight += weight;

        // Backward sample
        samplePos = tex - offset;
        sampleDepth = tex2D(sDepth, samplePos).r;

        depthWeight = softDepthCompare(centerDepth, sampleDepth);
        weight = depthWeight * coneWeight;

        color += tex2D(sPass, samplePos) * weight;
        totalWeight += weight;
    }

    return color / totalWeight;
}

technique T0 < string MGEinterface = "MGE XE 0"; >
{
    pass { PixelShader = compile ps_3_0 Mask(); }
    pass { PixelShader = compile ps_3_0 ObjectMotionBlur(); }
}
