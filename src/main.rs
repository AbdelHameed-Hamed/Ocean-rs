mod math;
mod obj_loader;
mod vk_engine;
mod vk_initializers;

fn main() {
    let mut engine = vk_engine::VkEngine::new(1280, 720);

    engine.run();

    unsafe {
        engine.cleanup();
    };
}
