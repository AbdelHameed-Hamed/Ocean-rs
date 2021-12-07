struct VSIn {
    float2 pos: Position;
    float2 uv: TexCoord;
    float4 col: Color;
};

struct VSOut {
    float4 pos: SV_Position;
    float4 col: Color;
    float2 uv: TexCoord;
};

[[vk::push_constant]]
struct {
    float2 scale;
    float2 translate;
} transfrom;

VSOut vs_main(VSIn input) {
    VSOut output = {
        float4(input.pos * transfrom.scale + transfrom.translate, 0, 1),
        input.col,
        input.uv
    };
    return output;
}

Texture2D<float4> tex: register(t0, space0);
sampler sam: register(s0, space0);

float4 fs_main(VSOut input): SV_Target {
    return tex.Sample(sam, input.uv) * input.col;
}