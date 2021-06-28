use std::f32::consts::PI;

// Reference: https://en.wikipedia.org/wiki/Xorshift
pub fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Reference: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
pub fn box_muller_rng(u1: f32, u2: f32) -> (f32, f32) {
    assert!(u1 < 1.0 && u1 > 0.0 && u2 < 1.0 && u2 > 0.0);

    let a = f32::sqrt(-2.0 * f32::ln(u1));
    let b = 2.0 * PI * u2;

    return (a * f32::cos(b), a * f32::sin(b));
}
