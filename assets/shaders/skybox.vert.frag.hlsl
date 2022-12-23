struct VSOut {
    float4 pos: SV_Position;
};

VSOut vs_main(uint vertex_idx: SV_VertexID) {
    float4 quad_verts[] = {
        float4(-1.0,-1.0, 0.0, 1.0),
        float4(-1.0, 1.0, 0.0, 1.0),
        float4( 1.0, 1.0, 0.0, 1.0),

        float4(-1.0,-1.0, 0.0, 1.0),
        float4( 1.0, 1.0, 0.0, 1.0),
        float4( 1.0,-1.0, 0.0, 1.0),
    };

    return VSOut(quad_verts[vertex_idx]);
}

cbuffer SceneData: register(b0, space0) {
    row_major float4x4 view; // This isn't right.....
    column_major float4x4 projection;
    float4 fog_color;
    float4 fog_distances;
    float4 ambient_color;
    float4 sunlight_direction;
    float4 sunlight_color;
};

float4 fs_main(float4 frag_coord: SV_Position): SV_Target {
    float2 screen_size = ambient_color.xy;

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

    float3 col;

    // Ground
    if (rd.y < 0.0) {
        col = float3(0.42, 0.39, 0.36);
    } else {
        // Sky with haze
        col = float3(0.3, 0.55, 0.8) * (1.0 - 0.8 * rd.y) * 0.9;

        // Sun
        float sundot = clamp(dot(rd.xyz, normalize(float3(-0.8, 0.3, -0.3))), 0.0, 1.0);
        col += 0.25 * float3(1.0, 0.7, 0.4) * pow(sundot, 8.0);
        col += 0.75 * float3(1.0, 0.8, 0.5) * pow(sundot, 64.0);

        // float v = smoothstep(-.05, 0.02, rd.y);
        // col = lerp(col, pow(col, float3(0.4545, 0.0, 0.0)), v);
    }

    // Horizon/atmospheric perspective
    col = lerp(col, float3(0.7, 0.75, 0.8), pow(1.0 - max(abs(rd.y), 0.0), 24.0));

    return float4(col, 1.0);
}