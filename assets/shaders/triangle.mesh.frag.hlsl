cbuffer SceneData: register(b0) {
    column_major float4x4 view;
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, zw unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

struct Vertex {
    float3 pos;
    float3 norm;
};

struct Meshlet {
    uint32_t vertex_count;
    uint32_t vertex_offset;
    uint32_t primitive_count;
    uint32_t primitive_offset;
};

StructuredBuffer<Meshlet> meshlets: register(t0, space1);
StructuredBuffer<Vertex> vertices: register(t1, space1);
StructuredBuffer<uint32_t> vertex_indices: register(t2, space1);
StructuredBuffer<uint32_t> primitive_indices: register(t3, space1);

struct OutputVertex {
    float4 pos: SV_Position;
    float4 col: Color;
};

[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void ms_main(
    in uint32_t group_id: SV_GroupID,
    in uint32_t group_thread_id: SV_GroupThreadID,
    out vertices OutputVertex out_verts[128],
    out indices uint32_t3 out_indices[128]) {
    Meshlet meshlet = meshlets[group_id];

    SetMeshOutputCounts(meshlet.vertex_count, meshlet.primitive_count);

    if (group_thread_id < meshlet.vertex_count) {
        uint32_t vertex_idx = vertex_indices[meshlet.vertex_offset + group_thread_id];
        Vertex vertex = vertices[vertex_idx];

        out_verts[group_thread_id].pos = mul(mul(projection, view), float4(vertex.pos, 1.0f));
        out_verts[group_thread_id].col = float4(vertex.norm, 1.0f);
    }

    if (group_thread_id < meshlet.primitive_count) {
        uint32_t packed_indices = primitive_indices[meshlet.primitive_offset + group_thread_id];

        out_indices[group_thread_id] = uint32_t3(
             packed_indices        & 0xFF,
            (packed_indices >> 8)  & 0xFF,
            (packed_indices >> 16) & 0xFF
        );
    }
}

float4 fs_main(OutputVertex input): SV_Target {
    return input.col + ambient_color;
}
