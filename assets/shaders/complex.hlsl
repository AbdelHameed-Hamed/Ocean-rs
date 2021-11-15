#ifndef __COMPLEX__
#define __COMPLEX__

#define TWO_PI 6.283185307179586476925286766559

float2 complex_mul(float2 lhs, float2 rhs) {
    float2 result = {
        lhs.x * rhs.x - lhs.y * rhs.y,
        lhs.x * rhs.y + lhs.y * rhs.x
    };

    return result;
}

float2 complex_exp(float imag) {
    float2 result = { cos(imag), sin(imag) };

    return result;
}

float2 complex_add(float2 lhs, float2 rhs) {
    float2 result = { lhs.x + rhs.x, lhs.y + rhs.y };

    return result;
}

float2 complex_float_mul(float2 lhs, float rhs) {
    float2 result = { lhs.x * rhs, lhs.y * rhs };

    return result;
}

float2 complex_conjugate(float2 self) {
    float2 result = { self.x, -self.y };

    return result;
}

#endif