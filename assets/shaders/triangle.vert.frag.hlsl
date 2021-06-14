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
    column_major float4x4 model_matrix;
} model_data;

cbuffer transforms2: register(b0) {
    column_major float4x4 view;
    column_major float4x4 projection;
};

VSOut vs_main(VSIn input) {
    VSOut output = {
        mul(
            mul(projection, mul(view, model_data.model_matrix)),
            float4(input.position, 1.0f)
        ),
        float4(input.color, 1.0f)
    };
    return output;
}

float4 fs_main(VSOut input) : SV_TARGET {
    return input.color;
}
