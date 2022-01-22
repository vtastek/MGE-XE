// Motion Blur
// by Knu
// adapted to MGE XE by Dexter

// **
// ** ADJUSTABLE VARIABLES

#define N 10 // Number of samples. Affects overall quality (smoothness) of the effect and performance.

static float mult_rot = 0.75 / 100.0; // Multiplier for rotational blur.
static float mult_mov = 2.00 / 100.0; // Multiplier for movement blur.
static float mask_distance = 33.0; // Set higher, if your hands/weapon get blurred.
static float max_blur = 10.0; // Maximum blur about in % of screen width.

// ** END OF
// **

float fov;
float frametime = 33.0/1000.0;
float4x4 mviewInv;
float4x4 mviewLast;
float2 rcpres;

static float mult_now = mult_rot / mult_mov;
static float2 t = 2.0 * tan(radians(fov * 0.5)) * float2( 1.0, -rcpres.x / rcpres.y );
static float sky = 100000;
static float2 raspect = rcpres.x / rcpres;
static float max_blur_m = max_blur / 100.0;

texture lastshader;
texture lastpass;
texture depthframe;

sampler sDepth = sampler_state { texture = <depthframe>; addressu = clamp; addressv = clamp; magfilter = point; minfilter = point; };

sampler sFrame = sampler_state { texture = < lastshader >; magfilter = point; minfilter = point; };
sampler sPass = sampler_state { texture = < lastpass >; magfilter = point; minfilter = point; addressu = mirror; addressv = mirror; };
sampler sPassLinear = sampler_state { texture = < lastpass >; magfilter = linear; minfilter = linear; addressu = border; addressv = border; bordercolor = float4(0.0, 0.0, 0.0, 0.0); };

float3 toView(float2 tex, float depth)
{
    float2 xy = (tex - 0.5) * depth * t;
    return float3(xy, depth);
}

float2 fromView(float3 view)
{
    return view / t / view.z + 0.5;
}
 float4 Mask( in float2 tex : TEXCOORD ) : COLOR0
{
    float mask = ( tex2D(sDepth, tex).r > mask_distance );
    return mask ? tex2D( sFrame, tex ) : 0;
}
float4 MotionBlur( in float2 tex : TEXCOORD ) : COLOR0
{
	
    float depth = min(tex2D(sDepth, tex).r, sky);
    float mask = ( depth > mask_distance );

    float4 now = float4( toView( tex, depth ) * mult_now, 1.0 );
    float4 then = mul( mul( now, mviewInv ), mviewLast );
    float2 motion = tex - fromView( then );
    float m = length(motion * raspect);
    m = min( m, max_blur_m ) / m / frametime * mult_rot;
    motion *= m;
    float2 s_tex = tex - motion;
    motion /= float( N );

    float4 color = 0;
	if(mask) {
    for (int i = 0; i <= 2 * N; i++)
    {
        color += pow(tex2D( sPass, s_tex ),2.2);
        s_tex += motion;   
    }
    color /= float(N * 2 + 1);
	}
	//return float4(motion*10, 0, 1);
	return depth > mask_distance * 1.1 ? float4(pow(color.xyz, 1.0/2.2),1.0) : tex2D( sFrame, tex );
    //return float4(pow(color.xyz, 1.0/2.2), 1);
}

technique T0 < string MGEinterface = "MGE XE 0"; >
{
    pass { PixelShader = compile ps_3_0 Mask(); }
    pass { PixelShader = compile ps_3_0 MotionBlur(); }
}
