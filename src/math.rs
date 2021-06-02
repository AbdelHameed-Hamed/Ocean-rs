use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    pub x: f32, pub y: f32, pub z: f32,
}

impl Vec3 {
    pub fn new(data: f32) -> Vec3 {
        return Vec3 { x: data, y: data, z: data };
    }

    pub fn up() -> Vec3 {
        return Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    }

    pub fn dot(lhs: Vec3, rhs: Vec3) -> f32 {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    pub fn cross(lhs: Vec3, rhs: Vec3) -> Vec3 {
        return Vec3 {
            x: lhs.y * rhs.z - lhs.z * rhs.y,
            y: lhs.z * rhs.x - lhs.x * rhs.z,
            z: lhs.x * rhs.y - lhs.y * rhs.x,
        };
    }

    pub fn length(self) -> f32 {
        return Self::dot(self, self).sqrt();
    }

    pub fn normalize(mut self) {
        self /= self.length();
    }

    pub fn normal(self) -> Vec3 {
        return self / self.length();
    }
}

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        return Vec3 { x: -self.x, y: -self.y, z: -self.z };
    }
}

macro_rules! vec3_vec3_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<Vec3> for Vec3 {
            type Output = Vec3;

            fn $fn(self, rhs: Vec3) -> Self::Output {
                return Vec3 {
                    x: self.x.$fn(rhs.x),
                    y: self.y.$fn(rhs.y),
                    z: self.z.$fn(rhs.z),
                };
            }
        }
    };
}

macro_rules! vec3_vec3_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<Vec3> for Vec3 {
            fn $fn(&mut self, rhs: Vec3) {
                self.x.$fn(rhs.x);
                self.y.$fn(rhs.y);
                self.z.$fn(rhs.z);
            }
        }
    };
}

macro_rules! vec3_f32_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Vec3 {
            type Output = Vec3;

            fn $fn(self, rhs: f32) -> Self::Output {
                return Vec3 { x: self.x.$fn(rhs), y: self.y.$fn(rhs), z: self.z.$fn(rhs) };
            }
        }

        impl $op<Vec3> for f32 {
            type Output = Vec3;

            fn $fn(self, rhs: Vec3) -> Self::Output {
                return Vec3 { x: rhs.x.$fn(self), y: rhs.y.$fn(self), z: rhs.z.$fn(self) };
            }
        }
    };
}

macro_rules! vec3_f32_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Vec3 {
            fn $fn(&mut self, rhs: f32) {
                self.x.$fn(rhs);
                self.y.$fn(rhs);
                self.z.$fn(rhs);
            }
        }
    };
}

macro_rules! vec3_ops {
    () => {
        vec3_vec3_binary_op!(Add, add);
        vec3_vec3_binary_op!(Sub, sub);
        vec3_vec3_opassign!(AddAssign, add_assign);
        vec3_vec3_opassign!(SubAssign, sub_assign);
        vec3_f32_binary_op!(Mul, mul);
        vec3_f32_binary_op!(Div, div);
        vec3_f32_opassign!(MulAssign, mul_assign);
        vec3_f32_opassign!(DivAssign, div_assign);
    };
}

vec3_ops!();

#[derive(Copy, Clone, Debug)]
pub struct Vec4 {
    pub x: f32, pub y: f32, pub z: f32, pub w: f32,
}

impl Vec4 {
    pub fn new(data: f32) -> Vec4 {
        return Vec4 { x: data, y: data, z: data, w: data };
    }

    pub fn from_vec3(data: Vec3, w: f32) -> Vec4 {
        return Vec4 { x: data.x, y: data.y, z: data.z, w: w };
    }

    pub fn dot(lhs: Vec4, rhs: Vec4) -> f32 {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
    }

    pub fn length(self) -> f32 {
        return Self::dot(self, self).sqrt();
    }

    pub fn normalize(mut self) {
        self /= self.length();
    }

    pub fn normal(self) -> Vec4 {
        let res = self.clone();
        res.normalize();
        return res;
    }
}

impl Neg for Vec4 {
    type Output = Vec4;

    fn neg(self) -> Self::Output {
        return Vec4 { x: -self.x, y: -self.y, z: -self.z, w: -self.w };
    }
}

macro_rules! vec4_vec4_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<Vec4> for Vec4 {
            type Output = Vec4;

            fn $fn(self, rhs: Vec4) -> Self::Output {
                return Vec4 {
                    x: self.x.$fn(rhs.x),
                    y: self.y.$fn(rhs.y),
                    z: self.z.$fn(rhs.z),
                    w: self.w.$fn(rhs.w),
                };
            }
        }
    };
}

macro_rules! vec4_vec4_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<Vec4> for Vec4 {
            fn $fn(&mut self, rhs: Vec4) {
                self.x.$fn(rhs.x);
                self.y.$fn(rhs.y);
                self.z.$fn(rhs.z);
                self.w.$fn(rhs.w);
            }
        }
    };
}

macro_rules! vec4_f32_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Vec4 {
            type Output = Vec4;

            fn $fn(self, rhs: f32) -> Self::Output {
                return Vec4 {
                    x: self.x.$fn(rhs),
                    y: self.y.$fn(rhs),
                    z: self.z.$fn(rhs),
                    w: self.w.$fn(rhs),
                };
            }
        }

        impl $op<Vec4> for f32 {
            type Output = Vec4;

            fn $fn(self, rhs: Vec4) -> Self::Output {
                return Vec4 {
                    x: rhs.x.$fn(self),
                    y: rhs.y.$fn(self),
                    z: rhs.z.$fn(self),
                    w: rhs.w.$fn(self),
                };
            }
        }
    };
}

macro_rules! vec4_f32_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Vec4 {
            fn $fn(&mut self, rhs: f32) {
                self.x.$fn(rhs);
                self.y.$fn(rhs);
                self.z.$fn(rhs);
                self.w.$fn(rhs);
            }
        }
    };
}

macro_rules! vec4_ops {
    () => {
        vec4_vec4_binary_op!(Add, add);
        vec4_vec4_binary_op!(Sub, sub);
        vec4_vec4_opassign!(AddAssign, add_assign);
        vec4_vec4_opassign!(SubAssign, sub_assign);
        vec4_f32_binary_op!(Mul, mul);
        vec4_f32_binary_op!(Div, div);
        vec4_f32_opassign!(MulAssign, mul_assign);
        vec4_f32_opassign!(DivAssign, div_assign);
    };
}

vec4_ops!();

#[derive(Copy, Clone, Debug)]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}

impl Mat4 {
    pub fn new(data: f32) -> Mat4 {
        return Mat4 {
            cols: [
                Vec4::new(data),
                Vec4::new(data),
                Vec4::new(data),
                Vec4::new(data),
            ],
        };
    }

    pub fn identity() -> Mat4 {
        return Mat4 {
            cols: [
                Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
                Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
                Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
                Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            ],
        };
    }

    pub fn scale(scaling_amount: Vec3) -> Mat4 {
        let mut res = Self::identity();
        res.cols[0].x = scaling_amount.x;
        res.cols[1].y = scaling_amount.y;
        res.cols[2].z = scaling_amount.z;

        return res;
    }

    pub fn translate(translation_amount: Vec3) -> Mat4 {
        let mut res = Self::identity();
        res.cols[3] = Vec4::from_vec3(translation_amount, 1.0);

        return res;
    }

    // Probably wanna do rotations using quaternions at some point...
    // Reference: https://learnopengl.com/Getting-started/Transformations
    pub fn rotate(axis: Vec3, angle: f32) -> Mat4 {
        axis.normalize();

        let (a, b, c) = (angle.cos(), angle.sin(), 1.0 - angle.cos());

        return Mat4 {
            cols: [
                Vec4 {
                    x: a + axis.x.powi(2) * c,
                    y: axis.y * axis.x * c + axis.z * b,
                    z: axis.z * axis.x * c - axis.y * b,
                    w: 0.0
                },
                Vec4 {
                    x: axis.x * axis.y * c - axis.z * b,
                    y: a + axis.y.powi(2) * c,
                    z: axis.z * axis.y * c + axis.x * b,
                    w: 0.0
                },
                Vec4 {
                    x: axis.x * axis.z * c + axis.y * b,
                    y: axis.y * axis.z * c - axis.x * b,
                    z: a + axis.z.powi(2) * c,
                    w: 0.0
                },
                Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            ]
        };
    }

    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
        let mut res = Mat4::new(0.0);

        let front = (center - eye).normal();
        let right = Vec3::cross(front, up).normal();
        let up = Vec3::cross(right, front);

        // The 3 basis of the view space
        res.cols[0].x = right.x;
        res.cols[0].y = up.x;
        res.cols[0].z = -front.x;

        res.cols[1].x = right.y;
        res.cols[1].y = up.y;
        res.cols[1].z = -front.y;

        res.cols[2].x = right.z;
        res.cols[2].y = up.z;
        res.cols[2].z = -front.z;

        // The translation part
        res.cols[3].x = -Vec3::dot(right, eye);
        res.cols[3].y = -Vec3::dot(up, eye);
        res.cols[3].z = Vec3::dot(front, eye);
        res.cols[3].w = 1.0;

        return res;
    }

    pub fn prespective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let mut res = Mat4::identity();

        let t = (fovy * (std::f32::consts::PI / 360.0)).tan();

        res.cols[0].x = 1.0 / (aspect * t);
        res.cols[1].y = 1.0 / t;
        res.cols[2].w = -1.0;
        res.cols[2].z = (near + far) / (near - far);
        res.cols[3].z = (2.0 * near * far) / (near - far);
        res.cols[3].w = 0.0;

        return res;
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;

    fn mul(self, rhs: Mat4) -> Self::Output {
        return Mat4 {
            cols: [
                self.cols[0] * rhs.cols[0].x + self.cols[1] * rhs.cols[0].y +
                    self.cols[2] * rhs.cols[0].z + self.cols[3] * rhs.cols[0].w,
                self.cols[0] * rhs.cols[1].x + self.cols[1] * rhs.cols[1].y +
                    self.cols[2] * rhs.cols[1].z + self.cols[3] * rhs.cols[1].w,
                self.cols[0] * rhs.cols[2].x + self.cols[1] * rhs.cols[2].y +
                    self.cols[2] * rhs.cols[2].z + self.cols[3] * rhs.cols[2].w,
                self.cols[0] * rhs.cols[3].x + self.cols[1] * rhs.cols[3].y +
                    self.cols[2] * rhs.cols[3].z + self.cols[3] * rhs.cols[3].w,
            ]
        };
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Self::Output {
        return self.cols[0] * rhs.x + self.cols[1] * rhs.y + self.cols[2] * rhs.z + self.cols[3] * rhs.w;
    }
}

#[derive(Copy, Clone, Debug)]
struct Quat {
    real: f32,
    complex: Vec3,
}

impl Quat {
    pub fn conjugate(self) -> Quat {
        return Quat {
            real: self.real,
            complex: -self.complex,
        };
    }

    pub fn norm_squared(self) -> f32 {
        return (self * self.conjugate()).real;
    }

    pub fn norm(self) -> f32 {
        return self.norm_squared().sqrt();
    }

    pub fn normal(self) -> Quat {
        return self / self.norm();
    }

    pub fn inverse(self) -> Quat {
        return self.conjugate() / self.norm_squared();
    }

    pub fn dot(lhs: Quat, rhs: Quat) -> f32 {
        return lhs.real * rhs.real + Vec3::dot(lhs.complex, rhs.complex);
    }

    pub fn from_axis(axis: Vec3, angle_in_radians: f32) -> Quat {
        let real = (0.5 * angle_in_radians).cos();
        let complex = axis.normal() * (0.5 * angle_in_radians).sin();

        return Quat{ real, complex };
    }
}

impl Neg for Quat {
    type Output = Quat;

    fn neg(self) -> Self::Output {
        return Quat {
            real: -self.real,
            complex: -self.complex,
        };
    }
}

impl Mul<Quat> for Quat {
    type Output = Quat;

    fn mul(self, rhs: Quat) -> Self::Output {
        let (a, b) = (self.complex, rhs.complex);
        let c = self.real * b + rhs.real * a + Vec3::cross(a, b);

        return Quat {
            real: self.real * rhs.real - Vec3::dot(a, b),
            complex: c,
        };
    }
}

macro_rules! quat_quat_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<Quat> for Quat {
            type Output = Quat;

            fn $fn(self, rhs: Quat) -> Self::Output {
                return Quat {
                    real: self.real.$fn(rhs.real),
                    complex: self.complex.$fn(rhs.complex),
                };
            }
        }
    };
}

macro_rules! quat_quat_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<Quat> for Quat {
            fn $fn(&mut self, rhs: Quat) {
                self.real.$fn(rhs.real);
                self.complex.$fn(rhs.complex);
            }
        }
    };
}

macro_rules! quat_f32_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Quat {
            type Output = Quat;

            fn $fn(self, rhs: f32) -> Self::Output {
                return Quat {
                    real: self.real.$fn(rhs),
                    complex: self.complex.$fn(rhs),
                };
            }
        }

        impl $op<Quat> for f32 {
            type Output = Quat;

            fn $fn(self, rhs: Quat) -> Self::Output {
                return Quat {
                    real: self.$fn(rhs.real),
                    complex: self.$fn(rhs.complex),
                };
            }
        }
    };
}

macro_rules! quat_f32_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Quat {
            fn $fn(&mut self, rhs: f32) {
                self.real.$fn(rhs);
                self.complex.$fn(rhs);
            }
        }
    };
}

macro_rules! quat_ops {
    () => {
        quat_quat_binary_op!(Add, add);
        quat_quat_binary_op!(Sub, sub);
        quat_quat_opassign!(AddAssign, add_assign);
        quat_quat_opassign!(SubAssign, sub_assign);
        quat_f32_binary_op!(Mul, mul);
        quat_f32_binary_op!(Div, div);
        quat_f32_opassign!(MulAssign, mul_assign);
        quat_f32_opassign!(DivAssign, div_assign);
    };
}

quat_ops!();

// !Note: Should probably write a test suite at some point...
#[cfg(test)]
mod tests {
    #[test]
    fn testing_shit() {
    }
}
