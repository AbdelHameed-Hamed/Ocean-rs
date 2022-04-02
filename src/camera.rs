use crate::math::lin_alg::Vec3;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

#[derive(Debug)]
pub struct Camera {
    pub(crate) pos: Vec3,
    pub(crate) front: Vec3,
    pub(crate) up: Vec3,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) fov: f32,
    pub(crate) rotate_camera: bool,
}

impl Camera {
    pub fn handle_event(&mut self, event: &sdl2::event::Event, delta_time: f32) {
        match *event {
            Event::MouseMotion {
                xrel: x, yrel: y, ..
            } => {
                if self.rotate_camera == false {
                    return;
                }

                // Note: I'm not sure if xrel and yrel account for deltas between frames.
                let sensitivity = 0.1;
                self.yaw += sensitivity * x as f32;
                self.pitch = (self.pitch + sensitivity * y as f32).clamp(-89.0, 89.0);
                self.front = Vec3 {
                    x: self.yaw.to_radians().cos(),
                    y: self.pitch.to_radians().sin(),
                    z: self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
                };
            }
            Event::MouseWheel { y: scroll_y, .. } => {
                self.fov = (self.fov - scroll_y as f32).clamp(1.0, 60.0);
            }
            Event::KeyDown {
                keycode: Some(pressed_key),
                ..
            } => {
                let camera_speed = 100.0 * delta_time;
                if pressed_key == Keycode::W {
                    self.pos += self.front * camera_speed;
                }
                if pressed_key == Keycode::S {
                    self.pos -= self.front * camera_speed;
                }
                if pressed_key == Keycode::A {
                    self.pos -= Vec3::cross(self.front, self.up).normal() * camera_speed;
                }
                if pressed_key == Keycode::D {
                    self.pos += Vec3::cross(self.front, self.up).normal() * camera_speed;
                }
            }
            _ => {}
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        #[rustfmt::skip]
        let mut res = Camera {
            pos: Vec3 { x: -130.0, y: -130.0, z: 7.0 },
            front: Vec3 { x: 0.0, y: 0.0, z: -1.0 },
            up: Vec3::up(),
            yaw: 40.0,
            pitch: 30.0,
            fov: 45.0,
            rotate_camera: false,
        };
        res.front = Vec3 {
            x: res.yaw.to_radians().cos(),
            y: res.pitch.to_radians().sin(),
            z: res.yaw.to_radians().sin() * res.pitch.to_radians().cos(),
        };

        return res;
    }
}
