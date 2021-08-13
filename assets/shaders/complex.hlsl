#ifndef __COMPLEX__
#define __COMPLEX__

#define PI 3.14159265358979323846264338327950288

struct Complex {
    float real;
    float imag;
};

Complex complex_mul(Complex lhs, Complex rhs) {
    Complex result = {
        lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real
    };

    return result;
}

Complex complex_exp(float imag) {
    Complex result = { cos(imag), sin(imag) };

    return result;
}

Complex complex_add(Complex lhs, Complex rhs) {
    Complex result = { lhs.real + rhs.real, lhs.imag + rhs.imag };

    return result;
}

Complex complex_float_mul(Complex lhs, float rhs) {
    Complex result = { lhs.real * rhs, lhs.imag * rhs };

    return result;
}

#endif