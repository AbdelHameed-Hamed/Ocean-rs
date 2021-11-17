#include "complex.hlsl"

#define OCEAN_DIM 512
#define OCEAN_DIM_EXPONENT 9 // log_2(OCEAN_DIM)
#define OCEAN_DIM_RECIPROCAL 0.001953125 // 1 / OCEAN_DIM

cbuffer SceneData: register(b0, space0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, z for time, w is unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

Texture2D<float2> tilde_h_zero: register(t0, space1);
Texture2D<float> frequencies: register(t1, space1);

RWTexture2D<float4> ifft_output_input: register(u2, space1);
RWTexture2D<float4> ifft_input_output: register(u3, space1);

//------------------------------------------------------------------------------------------------------
// Compute Shader
//------------------------------------------------------------------------------------------------------

// Based on https://github.com/asylum2010/Asylum_Tutorials/blob/master/Media/ShadersGL/fourier_fft.comp
groupshared float2 pingpong[2][OCEAN_DIM];

[[vk::push_constant]]
struct {
    float4 flags;
} flags;

[numthreads(OCEAN_DIM, 1, 1)]
void cs_main(uint x: SV_GroupThreadID, uint z: SV_GroupID) {
    if (flags.flags.x == 0) {
        // Calculate spectrum
        uint2 loc1 = uint2(x, z);
        uint2 loc2 = uint2(OCEAN_DIM - x, OCEAN_DIM - z);

        float w_k_t = frequencies[loc1] * fog_distances.z;

        float2 tilde_h1 = tilde_h_zero[loc1];
        float2 tilde_h2 = tilde_h_zero[loc2] * float2(1, -1); // complex conjugation

        // Now we compute tilde_h at time t
        pingpong[0][x] = complex_add(
            complex_mul(tilde_h1, complex_exp(w_k_t)),
            complex_mul(tilde_h2, complex_exp(-w_k_t))
        );
    }

    // Do IFFT
    // STEP 1: load row/column and reorder
    int nj = (reversebits(x) >> (32 - OCEAN_DIM_EXPONENT)) & (OCEAN_DIM - 1);
    if (flags.flags.x == 0) {
        pingpong[1][nj] = pingpong[0][x];
    } else {
        pingpong[1][nj] = ifft_output_input[uint2(x, z)].xy;
    }

    GroupMemoryBarrierWithGroupSync();

    // STEP 2: perform butterfly passes
    int src = 1;

    for (int s = 1; s <= OCEAN_DIM_EXPONENT; ++s) {
        int m = 1L << s;            // butterfly group height
        int mh = m >> 1;            // butterfly group half height

        if (x % m < mh) {
            // twiddle factor W_N^k
            float2 w_n_k = complex_exp(TWO_PI * x / m);

            float2 even = pingpong[src][x];
            float2 odd = complex_mul(w_n_k, pingpong[src][x + mh]);

            pingpong[1 - src][x] = complex_add(even, odd);
            pingpong[1 - src][x + mh] = complex_add(even, complex_float_mul(odd, -1));
        }

        src = 1 - src;

        GroupMemoryBarrierWithGroupSync();
    }

    // STEP 3: write output
    uint2 idx = {z, x};
    float2 result = pingpong[src][x];
    if (flags.flags.x == 0) {
        ifft_output_input[idx].xy = result;
    } else {
        ifft_input_output[idx].xy = complex_float_mul(result, ((x + z) & 1) == 1 ? -1 : 1);
    }
}

//------------------------------------------------------------------------------------------------------
// Mesh Shader
//------------------------------------------------------------------------------------------------------

// Here we wanna generate a 16x16 ocean patch and figure out where its triangles and verts are gonna lie
// in world space, this'll generate 256 vertices and 450 triangles.
// Note: There's gonna be a bit of an overlap, this is necessary to ensure that the topology of the edge
// triangles is accounted for, i.e., without this overlap you'd have individual disconnected patches
// instead of one mega connected patch.
#define patch_dim 16
#define patch_vertex_count (patch_dim * patch_dim)
#define patch_triangle_count ((patch_dim - 1) * (patch_dim - 1) * 2)

struct OutputVertex {
    float4 pos: SV_Position;
};

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void ms_main(
    in uint group_thread_id: SV_GroupThreadID,
    in uint group_id: SV_GroupID,
    out vertices OutputVertex out_verts[patch_vertex_count],
    out indices uint3 out_tris[patch_triangle_count])
{
    SetMeshOutputCounts(patch_vertex_count, patch_triangle_count);

    uint group_idx_x = group_id % (OCEAN_DIM / patch_dim);
    uint group_idx_z = group_id / (OCEAN_DIM / patch_dim);

    // We start off by figuring where our vertices and triangles are, transform and register them.
    uint num_iterations = ceil(patch_vertex_count / 32.0);
    for (uint i = 0; i < num_iterations; ++i) {
        uint vert_idx = group_thread_id * num_iterations + i;
        if (vert_idx < patch_vertex_count) {
            uint x = vert_idx % patch_dim;
            uint z = vert_idx / patch_dim;

            uint global_x = group_idx_x * patch_dim + x + 1 - group_idx_x;
            uint global_z = group_idx_z * patch_dim + z + 1 - group_idx_z;
            uint2 global_idx = {(global_z - 1), (global_x - 1)};

            // Transform the vertex and register it
            out_verts[vert_idx].pos = mul(
                mul(projection, view),
                float4(global_x, ifft_input_output[global_idx].x * 15, global_z, 1.0)
            );

            // Now figure which quad you represent and register its triangles
            if (x < (patch_dim - 1) && z < (patch_dim - 1)) {
                uint quad_idx = z * (patch_dim - 1) + x;

                // Lower triangle, counter clockwise order
                out_tris[quad_idx * 2] = uint3(
                    vert_idx,                 // Upper left corner
                    vert_idx + patch_dim,     // Lower Left corner
                    vert_idx + patch_dim + 1  // Lower right corner
                );

                // Upper triangle, counter clockwise order
                out_tris[(quad_idx * 2) + 1] = uint3(
                    vert_idx,                 // Upper left corner
                    vert_idx + patch_dim + 1, // Lower right corner
                    vert_idx + 1              // Upper right corner
                );
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------
// Fragment Shader
//------------------------------------------------------------------------------------------------------

float4 fs_main(OutputVertex input): SV_Target {
    return float4(6/255.0, 66/255.0, 115/255.0, 1.0);
}
