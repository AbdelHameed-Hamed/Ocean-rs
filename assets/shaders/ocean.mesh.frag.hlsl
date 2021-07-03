//------------------------------------------------------------------------------------------------------
// Mesh Shader
//------------------------------------------------------------------------------------------------------

#include "complex.hlsl"

// Actually used
cbuffer SceneData: register(b0, space0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, z for time, w unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

// Unused
StructuredBuffer<Complex> tilde_h_zero: register(t0, space1);
StructuredBuffer<Complex> tilde_h_zero_conjugate: register(t1, space1);

// Actually used
StructuredBuffer<Complex> input: register(t2, space1);

struct OutputVertex {
    float4 pos: SV_Position;
};

// Here we wanna generate a 12x12 ocean patch and figure out where its triangles and verts are gonna lie
// in world space, this'll generate 144 vertices and 242 triangles.
#define patch_dim 16
#define patch_vertex_count (patch_dim * patch_dim)
#define patch_triangle_count ((patch_dim - 1) * (patch_dim - 1) * 2)

#define ocean_dim 512

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void ms_main(
    in uint group_thread_id: SV_GroupThreadID,
    in uint group_id: SV_GroupID,
    out vertices OutputVertex out_verts[patch_vertex_count],
    out indices uint3 out_tris[patch_triangle_count])
{
    SetMeshOutputCounts(patch_vertex_count, patch_triangle_count);

    // We start off by figuring where our vertices and triangles are, transform and register them.
    uint num_iterations = ceil(patch_vertex_count / 32.0);
    for (uint i = 0; i < num_iterations; ++i) {
        uint vert_idx = group_thread_id * num_iterations + i;
        if (vert_idx < patch_vertex_count) {
            uint x = vert_idx % patch_dim;
            uint z = vert_idx / patch_dim;

            uint global_x = (group_id % (ocean_dim / patch_dim)) * patch_dim + x + 1;
            uint global_z = (group_id / (ocean_dim / patch_dim)) * patch_dim + z + 1;
            uint global_idx = (global_z - 1) * ocean_dim + (global_x - 1);

            // Transform the vertex and register it
            out_verts[vert_idx].pos = mul(
                mul(projection, view),
                float4(global_x, input[global_idx].real * 1000.0, global_z, 1.0)
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
