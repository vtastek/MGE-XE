//============================================================================
// MGE XE - Forward SSAO prepass
// Mirrors SSAO Fast.fx closely so forward replay matches the post-process path.
//============================================================================

static const int SSAO_SAMPLES = 8;
static const int BLUR_TAPS = 8;
static const float SSAO_RADIUS = 12.0;
static const float SSAO_OCCLUSION_FALLOFF = 50.0;
static const float SSAO_BLUR_FALLOFF = 0.5;
static const float SSAO_BLUR_RADIUS = 5.0;
static const float SSAO_DEPTH_SCALE = 10000.0;
static const float SSAO_SKY_DEPTH = 1e6;
static const float SSAO_EPS = 1e-6;

float4 prepassParams : register(c0);
// xy = 1 / resolution, z = fov in degrees.

texture tex0 : register(t0);
texture tex1 : register(t1);

sampler samp0 : register(s0) = sampler_state {
    texture = <tex0>;
    minfilter = point;
    magfilter = point;
    mipfilter = none;
    addressu = clamp;
    addressv = clamp;
};

sampler samp1 : register(s1) = sampler_state {
    texture = <tex1>;
    minfilter = point;
    magfilter = point;
    mipfilter = none;
    addressu = wrap;
    addressv = wrap;
};

static const float3 dirs[SSAO_SAMPLES] = {
    float3(0.00762, -0.01247, 0.03311),
    float3(-0.61057, 0.20510, 0.58876),
    float3(0.55319, 0.67960, -0.19194),
    float3(-0.43533, 0.62404, 0.45133),
    float3(-0.02386, -0.03104, 0.01502),
    float3(-0.20990, 0.10082, 0.03849),
    float3(0.06331, -0.17620, -0.31359),
    float3(-0.12261, 0.00720, -0.12465)
};

static const float2 taps[BLUR_TAPS] = {
    float2(-0.695914, 0.457137), float2(-0.203345, 0.620716),
    float2(0.96234, -0.194983),  float2(0.473434, -0.480026),
    float2(0.507431, 0.064425),  float2(0.89642, 0.412458),
    float2(-0.32194, -0.932615), float2(-0.791559, -0.59771)
};

float4 sample0(float2 tex) {
    return tex2Dlod(samp0, float4(tex, 0, 0));
}

float2 rcpres() {
    return prepassParams.xy;
}

float2 invproj() {
    float halfFov = 0.5 * radians(prepassParams.z);
    return 2.0 * tan(halfFov) * float2(1.0, prepassParams.x / prepassParams.y);
}

float3 toView(float2 tex) {
    float depth = sample0(tex).r;
    float2 xy = depth * (tex - 0.5) * invproj();
    return float3(xy, depth);
}

float2 fromView(float3 view) {
    return (view.xy / max(view.z, SSAO_EPS)) / invproj() + float2(0.5, 0.5);
}

float2 pack2(float f) {
    return float2(f, frac(f * 255.0 - 0.5));
}

float unpack2(float2 f) {
    return f.x + ((f.y - 0.5) / 255.0);
}

float4 ps_forward_ssao(float2 tex : TEXCOORD0) : COLOR0 {
    float3 pos = toView(tex);

    if (pos.z <= 0.0 || pos.z > SSAO_SKY_DEPTH) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float3 left = pos - toView(tex + rcpres() * float2(-1.0, 0.0));
    float3 right = toView(tex + rcpres() * float2(1.0, 0.0)) - pos;
    float3 up = pos - toView(tex + rcpres() * float2(0.0, -1.0));
    float3 down = toView(tex + rcpres() * float2(0.0, 1.0)) - pos;

    float3 dx = length(left) < length(right) ? left : right;
    float3 dy = length(up) < length(down) ? up : down;

    float3 normal = normalize(cross(dy, dx));
    dy = normalize(cross(dx, normal));
    dx = normalize(dx);

    float3 rnd = tex2Dlod(samp1, float4(tex / rcpres() / 8.0, 0, 0)).xyz * 2.0 - 1.0;

    float AO = 0.0;
    float amount = 0.0;
    [unroll]
    for (int j = 0; j < SSAO_SAMPLES; ++j) {
        float3 ray = reflect(dirs[j] * SSAO_RADIUS, rnd);
        ray *= sign(ray.z);
        ray = dx * ray.x + dy * ray.y + normal * ray.z;

        float weight = dot(normalize(ray), normal);
        float3 occ = toView(fromView(pos + ray));
        float diff = (pos.z + ray.z) - 1.00025 * occ.z;

        amount += weight;
        AO += weight * step(0.0, diff) * exp2(-diff / SSAO_OCCLUSION_FALLOFF);
    }

    return float4(AO / max(amount, SSAO_EPS), pack2(pos.z / SSAO_DEPTH_SCALE), 1.0);
}

float4 ps_forward_ssao_blur(float2 tex : TEXCOORD0) : COLOR0 {
    float4 data = sample0(tex);
    float total = data.r;
    float depth = unpack2(data.gb);
    float rev = SSAO_BLUR_RADIUS * (2.0 * data.a - 1.0);
    float amount = 1.0;

    [unroll]
    for (int i = 0; i < BLUR_TAPS; ++i) {
        float2 sTex = tex + rcpres() * taps[i] * rev;
        float4 sData = sample0(sTex);
        float sDepth = unpack2(sData.gb);
        float weight = exp2(-abs(depth - sDepth) / max(depth, SSAO_EPS) / SSAO_BLUR_FALLOFF);

        amount += weight;
        total += sData.r * weight;
    }

    return float4(1.2 * total / amount, data.g, data.b, 1.0 - data.a);
}
