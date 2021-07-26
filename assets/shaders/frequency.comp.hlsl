#include "complex.hlsl"

cbuffer SceneData: register(b0, space0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, z for time, w unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

StructuredBuffer<Complex> tilde_h_zero: register(t0, space1);
StructuredBuffer<Complex> tilde_h_zero_conjugate: register(t1, space1);
RWStructuredBuffer<Complex> tilde_h_t: register(u2, space1);

#define ocean_dim 512
#define l_x 1000.0f
#define l_z 1000.0f
#define width 16
#define height 16

[numthreads(width, height, 1)]
void cs_main(in uint3 thread_id: SV_DispatchThreadID) {
    if (thread_id.x == 0 && thread_id.y == 0) {
        // Crashes DXC =)
        // printf("Time is: %f\n", fog_distances.z);
    }

    float2 k = uint2(thread_id.x * 2.0 * PI / l_x, thread_id.y * 2.0 * PI / l_z);
    float w_k_t = sqrt(9.81 * length(k)) * fog_distances.z;

    // For now I'm hardcoding the actual ocean patch width
    uint idx = thread_id.y * ocean_dim + thread_id.x;

    // Now we compute tilde_h at time t
    tilde_h_t[idx] = complex_add(
        complex_mul(tilde_h_zero[idx], complex_exp(w_k_t)),
        complex_mul(tilde_h_zero_conjugate[idx], complex_exp(-w_k_t))
    );
}