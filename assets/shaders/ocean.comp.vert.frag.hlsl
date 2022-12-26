// Reference:
// Ocean Surface Generation and Rendering - Thomas Gamper
// https://www.cg.tuwien.ac.at/research/publications/2018/GAMPER-2018-OSG/GAMPER-2018-OSG-thesis.pdf

#define TWO_PI 6.283185307179586476925286766559

float2 complex_mul(float2 lhs, float2 rhs) {
    float2 result = {
        lhs.x * rhs.x - lhs.y * rhs.y,
        lhs.x * rhs.y + lhs.y * rhs.x
    };

    return result;
}

float2 complex_exp(float imag) {
    float2 result = { cos(imag), sin(imag) };

    return result;
}

float2 complex_conjugate(float2 self) {
    float2 result = { self.x, -self.y };

    return result;
}

// ToDo: Gotta rename things in this struct since they don't reflect the actual uses
cbuffer SceneData: register(b0, space0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 camera_pos;
    float4 fog_distances; //x: u, y: L, z: time, w: fov
    float4 ambient_color;
    float4 sunlight_direction;
    float4 sunlight_color;
};

// xy: tilda_h at time 0, zw: wavenumber
Texture2D<float4> waves: register(t0, space1);

// This implementation of the FFT algorithm requires two storage spaces for any single FFT being done.
// This is because it first computes the FFT for every row, then the FFT for every column, which means
// we might have a data race if we're not careful.

// x: height, y: unused, zw: displacement in the xz directions
RWTexture2D<float4> displacement_output_input: register(u1, space1);
RWTexture2D<float4> displacement_input_output: register(u2, space1);

// xy: partial derivatives of the height with resepect to x and z
// z: partial derivative of x's displacement with respect to x
// w: partial derivative of z's displacement with respect to z
RWTexture2D<float4> derivatives_output_input: register(u3, space1);
RWTexture2D<float4> derivatives_input_output: register(u4, space1);

//------------------------------------------------------------------------------------------------------
// Compute Shader
//------------------------------------------------------------------------------------------------------

// Based on https://github.com/asylum2010/Asylum_Tutorials/blob/master/Media/ShadersGL/fourier_fft.comp
groupshared float4 displacement_pingpong[2][OCEAN_DIM];
groupshared float4 derivatives_pingpong[2][OCEAN_DIM];

[numthreads(OCEAN_DIM, 1, 1)]
void cs_main(uint x: SV_GroupThreadID, uint z: SV_GroupID) {
    #ifdef CALCULATE_SPECTRUM_AND_ROW_IFFT
        // Calculate spectrum
        uint2 loc1 = {x, z};
        uint2 loc2 = {OCEAN_DIM - x, OCEAN_DIM - z};

        float2 k = waves[loc1].zw;

        float w_k_t = sqrt(9.81 * length(k)) * fog_distances.z;

        // Calculate tilde_h at time t
        float2 tilde_h1 = waves[loc1].xy;
        float2 tilde_h2 = complex_conjugate(waves[loc2].xy);

        float2 tilde_h_t =
            complex_mul(tilde_h1, complex_exp(w_k_t)) +
            complex_mul(tilde_h2, complex_exp(-w_k_t));

        float2 d_t = complex_mul(k.yx / length(k) * float2(1, -1), tilde_h_t);
        if (length(k) <= 0.00001)
            d_t = float2(0, 0);

        // Calculate the displacement in the xyz directions
        displacement_pingpong[0][x].xy = tilde_h_t;
        displacement_pingpong[0][x].zw = d_t;

        // Calculate the partial derivatives of height with respect to xz at time t
        derivatives_pingpong[0][x].xy = complex_mul(k.yx * float2(-1, 1), tilde_h_t);

        // Calculate the partial derivatives of xz displacements with respect to xz at time t
        derivatives_pingpong[0][x].zw = complex_mul(k.yx * float2(-1, 1), d_t);
    #endif

    // Do IFFT
    // STEP 1: load row/column and reorder
    int nj = (reversebits(x) >> (32 - OCEAN_DIM_EXPONENT)) & (OCEAN_DIM - 1);
    #ifdef CALCULATE_SPECTRUM_AND_ROW_IFFT
        displacement_pingpong[1][nj] = displacement_pingpong[0][x];
        derivatives_pingpong[1][nj] = derivatives_pingpong[0][x];
    #else
        displacement_pingpong[1][nj] = displacement_output_input[uint2(x, z)];
        derivatives_pingpong[1][nj] = derivatives_output_input[uint2(x, z)];
    #endif

    GroupMemoryBarrierWithGroupSync();

    // STEP 2: perform butterfly passes
    int src = 1;

    [unroll(OCEAN_DIM_EXPONENT)]
    for (int s = 1; s <= OCEAN_DIM_EXPONENT; ++s) {
        int m = 1L << s;            // butterfly group displacement
        int mh = m >> 1;            // butterfly group half displacement

        if (x % m < mh) {
            // Twiddle factor W_N^k
            float2 w_n_k = complex_exp(TWO_PI * x / m);

            // Height calculations
            float2 even = displacement_pingpong[src][x].xy;
            float2 odd = complex_mul(w_n_k, displacement_pingpong[src][x + mh].xy);

            displacement_pingpong[1 - src][x].xy = even + odd;
            displacement_pingpong[1 - src][x + mh].xy = even - odd;

            // xz displacement calculations
            even = displacement_pingpong[src][x].zw;
            odd = complex_mul(w_n_k, displacement_pingpong[src][x + mh].zw);

            displacement_pingpong[1 - src][x].zw = even + odd;
            displacement_pingpong[1 - src][x + mh].zw = even - odd;

            // Partial derivative of height with respect to xz
            even = derivatives_pingpong[src][x].xy;
            odd = complex_mul(w_n_k, derivatives_pingpong[src][x + mh].xy);

            derivatives_pingpong[1 - src][x].xy = even + odd;
            derivatives_pingpong[1 - src][x + mh].xy = even - odd;

            // Partial derivative of xz displacement with respect to xz
            even = derivatives_pingpong[src][x].zw;
            odd = complex_mul(w_n_k, derivatives_pingpong[src][x + mh].zw);

            derivatives_pingpong[1 - src][x].zw = even + odd;
            derivatives_pingpong[1 - src][x + mh].zw = even - odd;
        }

        src = 1 - src;

        GroupMemoryBarrierWithGroupSync();
    }

    // STEP 3: write output
    // Transpose the image
    uint2 idx = {z, x};
    float4 displacement_result = displacement_pingpong[src][x];
    float4 derivatives_result = derivatives_pingpong[src][x];
    #ifdef CALCULATE_SPECTRUM_AND_ROW_IFFT
        displacement_output_input[idx] = displacement_result;
        derivatives_output_input[idx] = derivatives_result;
    #else
        float sign_correction = ((x + z) & 1) == 1 ? -1 : 1;
        displacement_input_output[idx] = (displacement_result * sign_correction).zxwy;
        derivatives_input_output[idx] = derivatives_result * sign_correction;
    #endif
}

//------------------------------------------------------------------------------------------------------
// Vertex Shader
//------------------------------------------------------------------------------------------------------

struct VSIn {
    float4 pos: SV_Position;
};

struct VSOut {
    float4 view_pos: SV_Position;
    float4 normal: Normal;
    float4 world_pos: Position;
};

VSOut vs_main(VSIn input) {
    const float u = fog_distances.x;

    VSOut output;
    output.view_pos = mul(
        mul(projection, view),
        float4(
            input.pos.x,
            displacement_input_output[input.pos.xz].y * 8,
            input.pos.z,
            1
        )
    );

    float3 normal = float3(
        -(
            derivatives_input_output[input.pos.xz].xy /
            (1 + u * derivatives_input_output[input.pos.xz].zw)
        ),
        1
    ).xzy;
    output.normal = float4(normalize(normal), 1);
    output.world_pos = float4(input.pos.x, displacement_input_output[input.pos.xz].y * 8, input.pos.z, 1);

    return output;
}

//------------------------------------------------------------------------------------------------------
// Fragment Shader
//------------------------------------------------------------------------------------------------------

float4 fs_main(VSOut input): SV_Target {
    float3 ocean_color = { 0.1812f, 0.4678f, 0.5520f };
    float3 light_color = { 1.0, 1.0, 1.0 };

    float ambient_strength = 0.3;
    float3 ambient_color = ambient_strength * light_color;

    float3 light_dir = normalize(float3(255, 10, 255) - input.world_pos.xyz);
    float diffuse_strength = max(dot(input.normal.xyz, light_dir), 0);
    float3 diffuse_color = diffuse_strength * light_color;

    float specular_strength = 0.5;
    float3 view_dir = normalize(camera_pos - input.world_pos);
    float3 reflect_dir = reflect(-light_dir, input.normal);
    float specular_intensity = pow(max(dot(view_dir, reflect_dir), 0), 32);
    float3 specular_color = specular_strength * specular_intensity * light_color;

    return float4((ambient_color + diffuse_color + specular_color) * ocean_color, 1);
}
