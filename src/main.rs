mod camera;
mod imgui_backend;
mod math;
mod obj_loader;
mod shader_pp;
mod vk_engine;
mod vk_helpers;

const TEST_STR: &str = "@Input {
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
}";

fn main() {
    let tokens = shader_pp::tokenize(TEST_STR).unwrap();
    let ast = shader_pp::parse(tokens).unwrap();

    shader_pp::print_ast(&ast);

    let mut engine = vk_engine::VkEngine::new(1280, 720);

    engine.run();
}
