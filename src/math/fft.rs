use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

const PI: f32 = 3.141592653589793238463;

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    real: f32,
    imag: f32,
}

impl Mul<Complex> for Complex {
    type Output = Complex;

    fn mul(self, other: Self) -> Self::Output {
        return Complex {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        };
    }
}

macro_rules! complex_complex_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<Complex> for Complex {
            type Output = Complex;

            fn $fn(self, rhs: Complex) -> Self::Output {
                return Complex {
                    real: self.real.$fn(rhs.real),
                    imag: self.imag.$fn(rhs.imag),
                };
            }
        }
    };
}

macro_rules! complex_complex_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<Complex> for Complex {
            fn $fn(&mut self, rhs: Complex) {
                self.real.$fn(rhs.real);
                self.imag.$fn(rhs.imag);
            }
        }
    };
}

macro_rules! complex_f32_binary_op {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Complex {
            type Output = Complex;

            fn $fn(self, rhs: f32) -> Self::Output {
                return Complex {
                    real: self.real.$fn(rhs),
                    imag: self.imag.$fn(rhs),
                };
            }
        }

        impl $op<Complex> for f32 {
            type Output = Complex;

            fn $fn(self, rhs: Complex) -> Self::Output {
                return Complex {
                    real: self.$fn(rhs.real),
                    imag: self.$fn(rhs.imag),
                };
            }
        }
    };
}

macro_rules! complex_f32_opassign {
    ($op:ident, $fn:ident) => {
        impl $op<f32> for Complex {
            fn $fn(&mut self, rhs: f32) {
                self.real.$fn(rhs);
                self.imag.$fn(rhs);
            }
        }
    };
}

macro_rules! complex_ops {
    () => {
        complex_complex_binary_op!(Add, add);
        complex_complex_binary_op!(Sub, sub);
        complex_f32_binary_op!(Mul, mul);
        complex_f32_binary_op!(Div, div);
        complex_complex_opassign!(AddAssign, add_assign);
        complex_complex_opassign!(SubAssign, sub_assign);
        complex_f32_opassign!(MulAssign, mul_assign);
        complex_f32_opassign!(DivAssign, div_assign);
    };
}

complex_ops!();

pub enum FFTDirection {
    Forward,
    Backward,
}

fn bit_reverse(input: &mut Vec<Complex>, offset: usize, length: usize) {
    let bits = (std::mem::size_of::<usize>() * 8) as u32 - length.leading_zeros() - 1;

    let mut reverse_index: usize;

    for i in 0..length / 2 {
        reverse_index = 0;
        for j in 0..bits {
            if (i >> j) & 1 > 0 {
                reverse_index |= 1 << (bits - j - 1);
            }
        }

        input.swap(i + offset, reverse_index + offset);
    }
}

// Reference: https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm
fn cooley_tukey_1d(input: &mut Vec<Complex>, direction: f32, offset: usize, length: usize) {
    bit_reverse(input, offset, length);

    let (mut W, mut even, mut odd): (Complex, Complex, Complex);

    // 1. stage (N = length = 8) take log => 3
    let mut k = 2;
    while k <= length {
        // 2. Do fft_1d for each stage
        // each stage is divided into (N / k) groups
        let mut j = 0;
        while j < length {
            // 3. Because of symmetry we do it together
            //  stage1:  stage2:       stage3:
            //  0, 1     0, 2 & 1, 3
            //  2, 3                   0,4 & 1,5 & 2,6 & 3,7
            //  4, 5     4, 6 & 5, 7
            //  6, 7
            let to = j + k / 2;
            for i in j..to {
                // Formula:
                // W^(k/N), k = The even of the DFT architecture of each group
                // stages:
                // 1- M = (N / 4) = 2
                // 2- M = (N / 2) = 4
                // 3- M = (N) = 8
                W = Complex {
                    real: f32::cos((2.0 * PI * i as f32) / k as f32),
                    imag: -direction * f32::sin((2.0 * PI * i as f32) / k as f32),
                };
                let idx = i + k / 2;

                // Even = DFT of even indexed part
                // Odd = W * DFT of odd indexed part
                even = input[i + offset];
                odd = W * input[idx + offset];

                // X[k]     = Even + W * Odd
                // X[k+N/2] = even - W * Odd
                input[i + offset] = even + odd;
                input[idx + offset] = even - odd;
            }

            j += k;
        }

        k <<= 1;
    }
}

fn fft_1d(input: &mut Vec<Complex>, offset: usize, length: usize) {
    assert!(usize::is_power_of_two(length) && input.len() >= (offset + length));
    cooley_tukey_1d(input, 1.0, offset, length);
}

fn ifft_1d(input: &mut Vec<Complex>, offset: usize, length: usize) {
    assert!(usize::is_power_of_two(length) && input.len() >= (offset + length));
    cooley_tukey_1d(input, -1.0, offset, length);

    for i in input.iter_mut() {
        *i /= length as f32;
    }
}

pub fn fft_ifft_2d(input: &mut Vec<Complex>, width: usize, height: usize, direction: FFTDirection) {
    assert!(
        input.len() == width * height
            && usize::is_power_of_two(width)
            && usize::is_power_of_two(height)
    );

    // First we do the rows
    for i in 0..height {
        let offset = i * width;
        match direction {
            FFTDirection::Forward => fft_1d(input, offset, width),
            FFTDirection::Backward => ifft_1d(input, offset, width),
        }
    }

    // Then we do the columns
    let mut temp_col: Vec<Complex> = vec![unsafe { std::mem::zeroed() }; height];
    for i in 0..width {
        let offset = i;
        let stride = width;
        for j in 0..height {
            temp_col[j] = input[offset + j * stride];
        }

        match direction {
            FFTDirection::Forward => fft_1d(&mut temp_col, 0, height),
            FFTDirection::Backward => ifft_1d(&mut temp_col, 0, height),
        }

        for j in 0..height {
            input[offset + j * stride] = temp_col[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reverse_test() {
        {
            #[rustfmt::skip]
            let mut input = vec![
                Complex { real: 20.0, imag: 0.0 },
                Complex { real: 10.0, imag: 0.0 },
                Complex { real:  5.0, imag: 0.0 },
                Complex { real:  2.5, imag: 0.0 },
            ];

            #[rustfmt::skip]
            let expected_output = vec![
                Complex { real: 20.0, imag: 0.0 }, // 00 -> 00 SAME
                Complex { real:  5.0, imag: 0.0 }, // 01 -> 10 2
                Complex { real: 10.0, imag: 0.0 }, // 10 -> 01 1
                Complex { real:  2.5, imag: 0.0 }, // 11 -> 11 SAME
            ];

            let length = input.len();
            bit_reverse(&mut input, 0, length);

            assert_eq!(input.len(), expected_output.len());
            for i in 0..input.len() {
                assert_eq!(input[i].real, expected_output[i].real);
                assert_eq!(input[i].imag, expected_output[i].imag);
            }
        }

        {
            #[rustfmt::skip]
            let mut input = vec![
                Complex { real: 320.0, imag: 0.0 },
                Complex { real: 160.0, imag: 0.0 },
                Complex { real:  80.0, imag: 0.0 },
                Complex { real:  40.0, imag: 0.0 },
                Complex { real:  20.0, imag: 0.0 },
                Complex { real:  10.0, imag: 0.0 },
                Complex { real:   5.0, imag: 0.0 },
                Complex { real:   2.5, imag: 0.0 },
            ];

            #[rustfmt::skip]
            let expected_output = vec![
                Complex { real: 320.0, imag: 0.0 }, // 000 -> 000 SAME
                Complex { real:  20.0, imag: 0.0 }, // 001 -> 100 4
                Complex { real:  80.0, imag: 0.0 }, // 010 -> 010 SAME
                Complex { real:   5.0, imag: 0.0 }, // 011 -> 110 6
                Complex { real: 160.0, imag: 0.0 }, // 100 -> 001 1
                Complex { real:  10.0, imag: 0.0 }, // 101 -> 101 SAME
                Complex { real:  40.0, imag: 0.0 }, // 110 -> 011 3
                Complex { real:   2.5, imag: 0.0 }, // 111 -> 111 SAME
            ];

            let length = input.len();
            bit_reverse(&mut input, 0, length);

            assert_eq!(input.len(), expected_output.len());
            for i in 0..input.len() {
                assert_eq!(input[i].real, expected_output[i].real);
                assert_eq!(input[i].imag, expected_output[i].imag);
            }
        }
    }

    fn almost_equal(a: f32, b: f32, max_diff: f32, max_ulp_diff: i32) -> bool {
        let diff = f32::abs(a - b);
        if diff <= max_diff {
            return true;
        }

        if (a < 0.0) != (b < 0.0) {
            return false;
        }

        let (ai, bi) = unsafe {
            (
                std::mem::transmute::<f32, i32>(a),
                std::mem::transmute::<f32, i32>(b),
            )
        };

        return i32::abs(ai - bi) <= max_ulp_diff;
    }

    #[test]
    fn fft_test() {
        {
            #[rustfmt::skip]
            let mut input = vec![
                Complex { real: 20.0, imag: 0.0 },
                Complex { real: 10.0, imag: 0.0 },
                Complex { real:  5.0, imag: 0.0 },
                Complex { real:  2.5, imag: 0.0 },
            ];

            #[rustfmt::skip]
            let expected_output = vec![
                Complex { real: 37.5, imag:  0.0 },
                Complex { real: 15.0, imag: -7.5 },
                Complex { real: 12.5, imag:  0.0 },
                Complex { real: 15.0, imag:  7.5 },
            ];

            let length = input.len();
            fft_1d(&mut input, 0, length);

            assert_eq!(input.len(), expected_output.len());
            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(input[i].real, expected_output[i].real, f32::EPSILON * 10.0, 1));
                assert!(almost_equal(input[i].imag, expected_output[i].imag, f32::EPSILON * 10.0, 1));
            };
        }

        {
            #[rustfmt::skip]
            let mut input = vec![
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
            ];

            #[rustfmt::skip]
            let expected_output = vec![
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
            ];

            let length = input.len();
            fft_1d(&mut input, 0, length);

            assert_eq!(input.len(), expected_output.len());
            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(input[i].real, expected_output[i].real, f32::EPSILON * 10.0, 1));
                assert!(almost_equal(input[i].imag, expected_output[i].imag, f32::EPSILON * 10.0, 1));
            };
        }

        {
            #[rustfmt::skip]
            let mut input = vec![
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
            ];

            #[rustfmt::skip]
            let expected_output = vec![
                Complex { real: 8.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
            ];

            let length = input.len();
            fft_1d(&mut input, 0, length);

            assert_eq!(input.len(), expected_output.len());
            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(input[i].real, expected_output[i].real, f32::EPSILON * 10.0, 1));
                assert!(almost_equal(input[i].imag, expected_output[i].imag, f32::EPSILON * 10.0, 1));
            };
        }

        {
            #[rustfmt::skip]
            let mut input = vec![
                Complex { real: 320.0, imag: 0.0 },
                Complex { real: 160.0, imag: 0.0 },
                Complex { real:  80.0, imag: 0.0 },
                Complex { real:  40.0, imag: 0.0 },
                Complex { real:  20.0, imag: 0.0 },
                Complex { real:  10.0, imag: 0.0 },
                Complex { real:   5.0, imag: 0.0 },
                Complex { real:   2.5, imag: 0.0 },
            ];

            #[rustfmt::skip]
            let expected_output = vec![
                Complex { real:      637.5, imag:         0.0 },
                Complex { real: 379.549513, imag: -207.582521 },
                Complex { real:      255.0, imag:      -127.5 },
                Complex { real: 220.450487, imag:  -57.582521 },
                Complex { real:      212.5, imag:         0.0 },
                Complex { real: 220.450487, imag:   57.582521 },
                Complex { real:      255.0, imag:       127.5 },
                Complex { real: 379.549513, imag:  207.582521 },
            ];

            let length = input.len();
            fft_1d(&mut input, 0, length);

            assert_eq!(input.len(), expected_output.len());
            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(input[i].real, expected_output[i].real, f32::EPSILON * 1000.0, 1));
                assert!(almost_equal(input[i].imag, expected_output[i].imag, f32::EPSILON * 1000.0, 1));
            };
        }
    }

    #[test]
    fn ifft_test() {
        {
            #[rustfmt::skip]
            let input = vec![
                Complex { real: 20.0, imag: 0.0 },
                Complex { real: 10.0, imag: 0.0 },
                Complex { real:  5.0, imag: 0.0 },
                Complex { real:  2.5, imag: 0.0 },
            ];

            let length = input.len();
            let mut result = input.clone();
            fft_1d(&mut result, 0, input.len());
            ifft_1d(&mut result, 0, input.len());

            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(result[i].real, input[i].real, f32::EPSILON * 10.0, 1));
                assert!(almost_equal(result[i].imag, input[i].imag, f32::EPSILON * 10.0, 1));
            };
        }

        {
            #[rustfmt::skip]
            let input = vec![
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
                Complex { real: 0.0, imag: 0.0 },
            ];

            let mut result = input.clone();
            fft_1d(&mut result, 0, input.len());
            ifft_1d(&mut result, 0, input.len());

            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(result[i].real, input[i].real, f32::EPSILON * 10.0, 1));
                assert!(almost_equal(result[i].imag, input[i].imag, f32::EPSILON * 10.0, 1));
            };
        }

        {
            #[rustfmt::skip]
            let input = vec![
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
                Complex { real: 1.0, imag: 0.0 },
            ];

            let mut result = input.clone();
            fft_1d(&mut result, 0, input.len());
            ifft_1d(&mut result, 0, input.len());

            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(result[i].real, input[i].real, f32::EPSILON * 10.0, 1));
                assert!(almost_equal(result[i].imag, input[i].imag, f32::EPSILON * 10.0, 1));
            };
        }

        {
            #[rustfmt::skip]
            let input = vec![
                Complex { real: 360.0, imag: 0.0 },
                Complex { real: 160.0, imag: 0.0 },
                Complex { real:  80.0, imag: 0.0 },
                Complex { real:  40.0, imag: 0.0 },
                Complex { real:  20.0, imag: 0.0 },
                Complex { real:  10.0, imag: 0.0 },
                Complex { real:   5.0, imag: 0.0 },
                Complex { real:   2.5, imag: 0.0 },
            ];

            let mut result = input.clone();
            fft_1d(&mut result, 0, input.len());
            ifft_1d(&mut result, 0, input.len());

            #[rustfmt::skip]
            for i in 0..input.len() {
                assert!(almost_equal(result[i].real, input[i].real, f32::EPSILON * 1000.0, 1));
                assert!(almost_equal(result[i].imag, input[i].imag, f32::EPSILON * 1000.0, 1));
            };
        }
    }

    #[test]
    fn fft_ifft_2d_test() {
        #[rustfmt::skip]
        let input = vec![
            Complex{ real:  1.0, imag: 0.0 },
            Complex{ real:  2.0, imag: 0.0 },
            Complex{ real:  3.0, imag: 0.0 },
            Complex{ real:  4.0, imag: 0.0 },
            Complex{ real:  5.0, imag: 0.0 },
            Complex{ real:  6.0, imag: 0.0 },
            Complex{ real:  7.0, imag: 0.0 },
            Complex{ real:  8.0, imag: 0.0 },
            Complex{ real:  9.0, imag: 0.0 },
            Complex{ real: 10.0, imag: 0.0 },
            Complex{ real: 11.0, imag: 0.0 },
            Complex{ real: 12.0, imag: 0.0 },
            Complex{ real: 13.0, imag: 0.0 },
            Complex{ real: 14.0, imag: 0.0 },
            Complex{ real: 15.0, imag: 0.0 },
            Complex{ real: 16.0, imag: 0.0 },
        ];

        #[rustfmt::skip]
        let mut expected_output = vec![
            Complex{ real: 136.0, imag:   0.0 },
            Complex{ real:  -8.0, imag:   8.0 },
            Complex{ real:  -8.0, imag:   0.0 },
            Complex{ real:  -8.0, imag:  -8.0 },
            Complex{ real: -32.0, imag:  32.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real: -32.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real: -32.0, imag: -32.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
            Complex{ real:   0.0, imag:   0.0 },
        ];

        let mut output = input.clone();
        fft_ifft_2d(&mut output, 4, 4, FFTDirection::Forward);

        #[rustfmt::skip]
        for i in 0..input.len() {
            println!("{}, {}", output[i].real, expected_output[i].real);
            println!("{}, {}\n", output[i].imag, expected_output[i].imag);
            assert!(almost_equal(output[i].real, expected_output[i].real, f32::EPSILON * 1000.0, 1));
            assert!(almost_equal(output[i].imag, expected_output[i].imag, f32::EPSILON * 1000.0, 1));
        };
    }
}
