struct VSIn {
    float3 position: POSITION;
    float3 color: COLOR;
};

struct VSOut {
    float4 position: SV_POSITION;
    float4 color: COLOR;
};

[[vk::push_constant]]
struct {
    column_major float4x4 model;
    column_major float4x4 view;
    column_major float4x4 projection;
} transforms;

VSOut vs_main(VSIn input) {
    VSOut output = {
        mul(
            mul(transforms.projection, mul(transforms.view, transforms.model)),
            float4(input.position, 1.0f)
        ),
        float4(input.color, 1.0f)
    };
    return output;
}

float4 fs_main(VSOut input) : SV_TARGET {
    return input.color;
}
