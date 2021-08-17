struct OutputVertex {
    float4 pos: SV_Position;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void ms_main(out vertices OutputVertex out_verts[4], out indices uint3 out_tris[2]) {
    SetMeshOutputCounts(4, 2);

    out_verts[0].pos = float4(-1.0,-1.0, 0.0, 1.0);
    out_verts[1].pos = float4(-1.0, 1.0, 0.0, 1.0);
    out_verts[2].pos = float4( 1.0, 1.0, 0.0, 1.0);
    out_verts[3].pos = float4( 1.0,-1.0, 0.0, 1.0);

    out_tris[0] = uint3(0, 1, 2);
    out_tris[1] = uint3(0, 2, 3);
}

cbuffer SceneData: register(b0, space0) {
    row_major float4x4 view; // This isn't right.....
    column_major float4x4 projection;
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, z for time, and w for FoV.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

float4 fs_main(float4 frag_coord: SV_Position): SV_Target {
    float2 screen_size = float2(1280, 720);

    // Camera position and direction.
    float fov = fog_distances.w * 3.1415927 / 180.0;
    float4 rd = float4(
        normalize(
            float3(
                frag_coord.xy - screen_size.xy * 0.5,
                screen_size.y * 0.5 / -tan(fov * 0.5)
            )
        ),
        1.0
    );
    rd *= -1;
    rd = mul(view, rd);

    // Sky with haze
    float3 col = float3(0.3, 0.55, 0.8) * (1.0 - 0.8 * rd.y) * 0.9;

    // Ground
    if(rd.y < 0.0)
        col = float3(0.42, 0.39, 0.36);

    // Horizon/atmospheric perspective
    col = lerp(col, float3(0.7, 0.75, 0.8), pow(1.0 - max(abs(rd.y), 0.0), 24.0));

    if(rd.y >= -0.05) {
        float v = smoothstep(-.05, 0.02, rd.y);
        col = lerp(col, pow(col, float3(0.4545, 0.0, 0.0)), v);
    }

    return float4(col, 1.0);
}