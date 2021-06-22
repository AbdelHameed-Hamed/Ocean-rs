cbuffer SceneData: register(b0, space0) {
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
    uint32_t vertices[64];
    uint16_t indices[126];
    uint16_t vertex_and_index_count;
};

StructuredBuffer<Meshlet> meshlets: register(t0, space1);
StructuredBuffer<Vertex> vertices: register(t1, space1);

struct OutputVertex {
    float4 pos: SV_Position;
    float4 col: Color;
};

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void ms_main(
    in uint32_t group_id: SV_GroupID,
    in uint32_t group_thread_id: SV_GroupThreadID,
    out vertices OutputVertex out_verts[64],
    out indices uint32_t3 out_primitives[42]) {
    Meshlet meshlet = meshlets[group_id];

    uint32_t vertex_count = meshlet.vertex_and_index_count & 0xFF;
    uint32_t index_count = meshlet.vertex_and_index_count >> 8;
    uint32_t primitive_count = index_count / 3;

    SetMeshOutputCounts(vertex_count, primitive_count);

    for (uint32_t offset = 0; offset < 2; ++offset) {
        uint32_t idx = group_thread_id + (offset * 32);
        if (idx < vertex_count) {
            uint32_t vertex_idx = meshlet.vertices[idx];
            Vertex vertex = vertices[vertex_idx];

            out_verts[idx].pos = mul(mul(projection, view), float4(vertex.pos, 1.0));
            out_verts[idx].col = float4(vertex.norm, 1.0);
        }

        if (idx < primitive_count) {
            out_primitives[idx] = uint32_t3(
                meshlet.indices[(idx * 3) + 0],
                meshlet.indices[(idx * 3) + 1],
                meshlet.indices[(idx * 3) + 2]
            );
        }
    }
}

float4 fs_main(OutputVertex input): SV_Target {
    return input.col + ambient_color;
}
