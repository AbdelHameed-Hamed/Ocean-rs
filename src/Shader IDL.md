# Example

```rs
@Input {
    waves: Tex2D[f32; 4],
    displacement_output_input: mut Tex2D[f32; 4],
    displacement_input_output: mut Tex2D[f32; 4],
    derivatives_output_input: mut Tex2D[f32; 4],
    derivatives_input_output: mut Tex2D[f32; 4],
    test_type: struct {
        x: f32,
        y: f32,
        z: f32,
    },
}
```
