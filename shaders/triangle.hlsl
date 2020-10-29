struct VSOut
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

VSOut VSMain(uint vertex_id: SV_VERTEXID)
{
    // just pass vertex position straight through
    const float3 positions[3] = {
        float3( 1.f, 1.f, 0.0f),
        float3(-1.f, 1.f, 0.0f),
        float3( 0.f,-1.f, 0.0f)
    };

    const float3 colors[3] = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)
    };

    VSOut output = {
        float4(positions[vertex_id], 1.0f),
        float4(colors[vertex_id], 1.0f)
    };
    return output;
}

float4 FSMain(VSOut input) : SV_TARGET
{
    return input.color;
}