mod camera;
mod imgui_backend;
mod math;
mod obj_loader;
mod vk_engine;
mod vk_helpers;

fn main() {
    let mut engine = vk_engine::VkEngine::new(1280, 720);

    engine.run();
}
