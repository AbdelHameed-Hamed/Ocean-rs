// Inverse fourier transform for each row in the buffer
// At some point I really should send in the dimensions of my buffer...

#include "complex.hlsl"

// Unused
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

// Actually used
RWStructuredBuffer<Complex> output: register(u1, space1);
StructuredBuffer<Complex> input: register(t2, space1);

#define ocean_dim 512
#define l_x 1000.0f
#define l_z 1000.0f
#define num_threads 64

[numthreads(num_threads, 1, 1)]
void cs_main(in uint3 thread_id: SV_DispatchThreadID) {
    uint output_idx = thread_id.y * ocean_dim + thread_id.x;
    Complex temp = { 0, 0 };
    output[output_idx] = temp;
    uint k = thread_id.x;
    for (uint j = 0; j < ocean_dim; ++j) {
        uint input_idx = thread_id.y * ocean_dim + j;
        float2 l = fog_distances.xy;
        output[output_idx] = complex_add(
            output[output_idx],
            complex_mul(
                input[input_idx],
                complex_exp(2 * PI * (j * 2.0 * PI / l.x) * (k * 2.0 * PI / l.y))
            )
        );
    }
    temp.real = 1 / float(ocean_dim);
    output[output_idx] = complex_mul(output[output_idx], temp);
}