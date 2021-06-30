cbuffer SceneData: register(b0, space0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, zw unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

struct OutputVertex {
    float4 pos: SV_Position;
};

// Here we wanna generate an 8x8 ocean patch and figure out where its triangles and verts are gonna lie
// in world space, this'll generate 81 vertices and 128 triangles.
#define patch_dim 8
#define patch_vertex_count ((patch_dim + 1) * (patch_dim + 1))
#define patch_triangle_count (patch_dim * patch_dim * 2)

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void ms_main(
    in uint group_id: SV_GroupID,
    in uint group_thread_id: SV_GroupThreadID,
    out vertices OutputVertex out_verts[patch_vertex_count],
    out indices uint3 out_tris[patch_triangle_count])
{
    SetMeshOutputCounts(patch_vertex_count, patch_triangle_count);

    for (uint i = 0; i < 2; ++i) {
        uint quad_idx = group_thread_id * 2 + i;

        // Index of this quad's upper left vertex
        uint upper_left_x = quad_idx % patch_dim;
        uint upper_left_y = quad_idx / patch_dim;
        uint upper_left_vert_idx = upper_left_y * (patch_dim + 1) + upper_left_x;

        // Transform the vertex and register it
        out_verts[upper_left_vert_idx].pos = mul(
            mul(projection, view),
            float4(upper_left_x, upper_left_y, 0.0, 1.0)
        );

        // Register the lower triangle
        out_tris[quad_idx * 2] = uint3(
            upper_left_vert_idx,
            upper_left_vert_idx + (patch_dim + 1),
            upper_left_vert_idx + (patch_dim + 1) + 1
        );

        // Register the upper triangle
        out_tris[(quad_idx * 2) + 1] = uint3(
            upper_left_vert_idx,
            upper_left_vert_idx + (patch_dim + 1) + 1,
            upper_left_vert_idx + 1
        );
    }

    // Leftover vertices on the bottom and right of the patch.
    // 81 - 64 = 17 vertex to be transformed and registered.
    if (group_thread_id < 17) {
        // I wonder if this worth avoiding the extra branch...
        uint temp = group_thread_id / 8;
        uint is_bottom = ((temp & 2) >> 1) ^ ((temp & 1) ^ 1);
        uint is_right = temp & 1;
        uint is_bottom_right = temp >> 1;

        // Crashes DXC?????????????????????????
        // if (is_bottom_right > 1) {
        //     printf("%d\n\n", group_thread_id);
        // }

        uint offset = group_thread_id % 8;
        uint x = (is_right * patch_dim) + (is_bottom *    offset) + (is_bottom_right * patch_dim);
        uint y = (is_right *    offset) + (is_bottom * patch_dim) + (is_bottom_right * patch_dim);
        uint vert_idx = y * (patch_dim + 1) + x;

        out_verts[vert_idx].pos = mul(mul(projection, view), float4(x, y, 0.0, 1.0));
    }
}

float4 fs_main(OutputVertex input): SV_Target {
    return float4(1.0, 1.0, 1.0, 1.0);
}
