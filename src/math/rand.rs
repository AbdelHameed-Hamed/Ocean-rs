use std::f32::consts::PI;

// Reference: https://github.com/imneme/pcg-c-basic
pub struct PCGRandom32 {
    state: u64,
    inc: u64,
}

impl PCGRandom32 {
    // Statically initialized seed, it's preferred to get the seed through the
    // OS at runtime instead.
    pub fn new() -> PCGRandom32 {
        return PCGRandom32 {
            state: 0x853c49e6748fea9b,
            inc: 0xda3e39cb94b95bdb,
        };
    }

    pub fn new_with_seed(state: u64, inc: u64) -> PCGRandom32 {
        return PCGRandom32 { state, inc };
    }

    // Get the next random number in a sequence
    pub fn next(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        return (xorshifted >> rot) | (xorshifted << (((!rot).wrapping_add(1)) & 31));
    }
}

// Reference: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
pub fn box_muller_rng(u1: f32, u2: f32) -> (f32, f32) {
    assert!(u1 < 1.0 && u1 > 0.0 && u2 < 1.0 && u2 > 0.0);

    let a = f32::sqrt(-2.0 * f32::ln(u1));
    let b = 2.0 * PI * u2;

    return (a * f32::cos(b), a * f32::sin(b));
}
