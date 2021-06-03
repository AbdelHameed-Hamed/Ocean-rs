struct VSIn {
    float3 position: POSITION;
    float3 color: COLOR;
};

struct VSOut {
    float4 position: SV_POSITION;
    float4 color: COLOR;
};

VSOut vs_main(VSIn input) {
    VSOut output = {
        float4(input.position, 1.0f),
        float4(input.color, 1.0f)
    };
    return output;
}

float4 fs_main(VSOut input) : SV_TARGET {
    return input.color;
}
