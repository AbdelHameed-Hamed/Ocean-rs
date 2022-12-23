RWStructuredBuffer<float4> vertices: register(u0, space0);
RWStructuredBuffer<uint> indices: register(u1, space0);

[numthreads(16, 16, 1)]
void create_ocean_grid(uint2 loc: SV_DispatchThreadID) {
    uint vert_idx = loc.y * OCEAN_DIM + loc.x;
    vertices[vert_idx] = float4(loc.x, 0, loc.y, 1);

    if (loc.x < (OCEAN_DIM - 1) && loc.y < (OCEAN_DIM - 1)) {
        uint quad_idx = loc.y * (OCEAN_DIM - 1) + loc.x;

        // Lower triangle in the quad.
        indices[(quad_idx * 6)]     = vert_idx;                 // Upper left corner
        indices[(quad_idx * 6) + 1] = vert_idx + OCEAN_DIM;     // Lower left corner
        indices[(quad_idx * 6) + 2] = vert_idx + OCEAN_DIM + 1; // Lower right corner

        // Upper triangle in the quad.
        indices[(quad_idx * 6) + 3] = vert_idx + OCEAN_DIM + 1; // Lower right corner
        indices[(quad_idx * 6) + 4] = vert_idx + 1;             // Upper right corner
        indices[(quad_idx * 6) + 5] = vert_idx;                 // Upper left corner
    }
}
