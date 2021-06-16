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

cbuffer Transforms: register(b0) {
    column_major float4x4 view;
    column_major float4x4 projection;
};

cbuffer SceneData: register(b1) {
    float4 fog_color; // w is for exponent
    float4 fog_distances; //x for min, y for max, zw unused.
    float4 ambient_color;
    float4 sunlight_direction; //w for sun power
    float4 sunlight_color;
};

struct Model {
    float4x4 model;
};

StructuredBuffer<Model> models: register(t0, space1);

VSOut vs_main(VSIn input, [[vk::builtin("BaseInstance")]] uint instance: GARBAGE) {
    VSOut output = {
        mul(
            mul(projection, mul(view, models[instance].model)),
            float4(input.position, 1.0f)
        ),
        float4(input.color, 1.0f)
    };
    return output;
}

float4 fs_main(VSOut input) : SV_TARGET {
    return input.color + ambient_color;
}
