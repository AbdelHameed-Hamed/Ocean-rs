cbuffer SceneData: register(b0, space0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, zw unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

struct Complex {
    float real;
    float imag;
};

Complex complex_mul(Complex lhs, Complex rhs) {
    Complex result = {
        lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real
    };

    return result;
}

Complex complex_exp(float imag) {
    Complex result = {
        cos(imag),
        sin(imag)
    };

    return result;
}

StructuredBuffer<Complex> tilde_h_zero: register(t0, space1);
StructuredBuffer<Complex> tilde_h_zero_conjugate: register(t1, space1);

struct OutputVertex {
    float4 pos: SV_Position;
};

// Here we wanna generate a 12x12 ocean patch and figure out where its triangles and verts are gonna lie
// in world space, this'll generate 144 vertices and 242 triangles.
#define patch_dim 12
#define patch_vertex_count (patch_dim * patch_dim)
#define patch_triangle_count ((patch_dim - 1) * (patch_dim - 1) * 2)

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void ms_main(
    in uint group_id: SV_GroupID,
    in uint group_thread_id: SV_GroupThreadID,
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

            // Transform the vertex and register it
            out_verts[vert_idx].pos = mul(mul(projection, view), float4(x + 1, 0.0, z + 1, 1.0));

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

float4 fs_main(OutputVertex input): SV_Target {
    return float4(1.0, 1.0, 1.0, 1.0);
}
