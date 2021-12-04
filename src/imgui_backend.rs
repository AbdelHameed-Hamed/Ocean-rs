pub fn handle_event(io: &mut imgui::Io, event: &sdl2::event::Event) {
    use sdl2::event::Event::*;
    use sdl2::mouse::*;

    match *event {
        MouseButtonDown { mouse_btn, .. } => match mouse_btn {
            MouseButton::Left => io.mouse_down[0] = true,
            MouseButton::Middle => io.mouse_down[2] = true,
            MouseButton::Right => io.mouse_down[1] = true,
            _ => {}
        },

        MouseButtonUp { mouse_btn, .. } => match mouse_btn {
            MouseButton::Left => io.mouse_down[0] = false,
            MouseButton::Middle => io.mouse_down[2] = false,
            MouseButton::Right => io.mouse_down[1] = false,
            _ => {}
        },

        MouseWheel { x, y, .. } => {
            io.mouse_wheel = y as f32;
            io.mouse_wheel_h = x as f32;
        }

        MouseMotion { x, y, .. } => io.mouse_pos = [x as f32, y as f32],

        _ => {}
    };
}
