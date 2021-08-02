#include "complex.hlsl"

#define OCEAN_DIM 512
#define OCEAN_DIM_EXPONENT 9
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

StructuredBuffer<Complex> tilde_h_zero: register(t0, space1);
StructuredBuffer<Complex> tilde_h_zero_conjugate: register(t1, space1);

RWStructuredBuffer<Complex> ifft_output: register(u2, space1);

//------------------------------------------------------------------------------------------------------
// Compute Shader
//------------------------------------------------------------------------------------------------------

// Based on https://github.com/asylum2010/Asylum_Tutorials/blob/master/Media/ShadersGL/fourier_fft.comp
groupshared Complex pingpong[2][OCEAN_DIM];

[[vk::push_constant]]
struct {
    float4 flags;
} flags;

[numthreads(OCEAN_DIM, 1, 1)]
void cs_main(uint x: SV_GroupThreadID, uint z: SV_GroupID) {
    if (flags.flags.x == 0) {
        // Calculate spectrum
        float2 l = fog_distances.xy;

        float2 k = uint2(x * 2.0 * PI / l.x, z * 2.0 * PI / l.y);
        float w_k_t = sqrt(9.81 * length(k)) * fog_distances.z;

        uint idx = z * OCEAN_DIM + x;

        // Now we compute tilde_h at time t
        pingpong[0][idx] = complex_add(
            complex_mul(tilde_h_zero[idx], complex_exp(w_k_t)),
            complex_mul(tilde_h_zero_conjugate[idx], complex_exp(-w_k_t))
        );

        GroupMemoryBarrierWithGroupSync();
    }

    // Do IFFT
    // STEP 1: load row/column and reorder
    int nj = (reversebits(x) >> (32 - OCEAN_DIM_EXPONENT)) & (OCEAN_DIM - 1);
    if (flags.flags.x == 0) {
        pingpong[1][nj] = pingpong[0][x];
    } else {
        pingpong[1][nj] = ifft_output[z * OCEAN_DIM + x];
    }

    GroupMemoryBarrierWithGroupSync();

    // STEP 2: perform butterfly passes
    int src = 1;

    for (int s = 1; s <= OCEAN_DIM_EXPONENT; ++s) {
        int m = 1U << s;            // butterfly group height
        int mh = m >> 1;            // butterfly group half height

        int k = (x * (OCEAN_DIM / m)) & (OCEAN_DIM - 1);
        int i = (x & ~(m - 1));     // butterfly group starting offset
        int j = (x & (mh - 1));     // butterfly index in group

        // twiddle factor W_N^k
        float theta = (2 * PI * float(k)) * OCEAN_DIM_RECIPROCAL;
        Complex W_N_k = { cos(theta), sin(theta) };

        Complex even = pingpong[src][i + j];
        Complex odd = pingpong[src][i + j + mh];

        src = 1 - src;
        pingpong[src][x] = complex_add(even, complex_mul(W_N_k, odd));

        GroupMemoryBarrierWithGroupSync();
    }

    // STEP 3: write output
    ifft_output[x * OCEAN_DIM + z].real = pingpong[src][x].real * OCEAN_DIM_RECIPROCAL;
    ifft_output[x * OCEAN_DIM + z].imag = pingpong[src][x].imag * OCEAN_DIM_RECIPROCAL;
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
            uint global_idx = (global_z - 1) * OCEAN_DIM + (global_x - 1);

            // Transform the vertex and register it
            out_verts[vert_idx].pos = mul(
                mul(projection, view),
                float4(global_x, ifft_output[global_idx].real, global_z, 1.0)
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
    return float4(1.0, 1.0, 1.0, 1.0);
}
