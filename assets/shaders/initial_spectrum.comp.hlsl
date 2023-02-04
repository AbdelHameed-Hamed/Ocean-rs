// Based on the paper : Empirical Directional Wave Spectra for Computer Graphics

static const float PI = 3.1415926535897932384626433832795;
static const float TWO_PI = 6.283185307179586476925286766559;
static const float g = 9.81;
static const float sigma = 0.074; // Surface tension coeffecient
static const float rho = 1000; // Density of water

[[vk::push_constant]] struct {
    float L;          // Ocean patch length
    float U;          // Wind speed
    uint F;           // Fetch
    float h;          // Ocean depth
    float wind_angle; // Wind angle with the x axis
    float swell;
    uint ocean_dim;   // Ocean patch dimension
    uint noise_tex_idx;
    uint waves_spectrum_idx;
} ocean_params;

Texture2D bindless_textures[]: register(t0, space0);
RWTexture2D<float4> bindless_rwtextures[]: register(u0, space1);

uint fetch() {
    return ocean_params.F * 1000;
}

// Eqn 9
float omega_value(float k) {
    return sqrt((g * k + sigma / rho * k * k * k) * tanh(min(k * ocean_params.h, 30)));
}

float omega_derivative(float k) {
    return ((g + 3 * sigma / rho * k * k) * tanh(min(k * ocean_params.h, 30)) +
        ocean_params.h * (g * k + sigma / rho * k * k * k) / cosh(min(k * ocean_params.h, 30)) / cosh(min(k * ocean_params.h, 30))) /
        omega_value(k);
}

float omega_peak(float omega) {
    return 22 * (g * g / (ocean_params.U * fetch()));
}

// Eqn 28
float jonswap(float omega) {
    float gamma = 3.3;
    float omega_p = omega_peak(omega);
    float sigma = (omega <= omega_p) ? 0.07 : 0.09;
    float r = exp(-(omega - omega_p) * (omega - omega_p) / (2 * sigma * sigma * omega_p * omega_p));
    float alpha = 0.076 * pow(ocean_params.U * ocean_params.U / (fetch() * g), 0.22);

    return alpha * g * g / (omega * omega * omega * omega * omega) * pow(gamma, r) *
        exp(-(5 / 4) * (omega_p / omega) * (omega_p / omega) * (omega_p / omega) * (omega_p / omega));
}

// Eqn 30
float tma(float omega) {
    float omega_h = omega * sqrt(ocean_params.h / g);
    float attenuation_factor;
    if (omega_h <= 1) {
        attenuation_factor = (omega_h * omega_h) / 2;
    } else if (omega_h < 2) {
        attenuation_factor = 1 - (2 - omega_h) * (2 - omega_h) / 2;
    } else {
        attenuation_factor = 1;
    }

    return jonswap(omega) * attenuation_factor;
}

// Eqn 38
float donelan_banner(float omega, float theta) {
    float omega_p = omega_peak(omega);
    float beta_s;
    if ((omega_p / omega_p) < 0.95) {
        beta_s = 2.61 * pow(omega / omega_p, 1.3);
    } else if ((omega / omega_p) < 1.6) {
        beta_s = 2.28 * pow(omega / omega_p, -1.3);
    } else {
        float epsilon = -0.4 + 0.8393 * exp(-0.567 * log((omega / omega_p) * (omega / omega_p)));
        beta_s = pow(10, epsilon);
    }

    return beta_s / (2 * tanh(PI * beta_s) * cosh(beta_s * theta) * cosh(beta_s * theta));
}

// Eqn 45
float s_xi(float omega) {
    return 16 * tanh(omega_peak(omega) / omega) * ocean_params.swell * ocean_params.swell;
}

// Eqn 44
float swell_factor(float omega, float theta) {
    return pow(abs(cos(theta / 2)), 2 * s_xi(omega));
}

// Eqn 43
float normalization_factor(float omega) {
    float denominator = 0;
    float step_size = 0.1;

    for (float i = -PI / 2; i <= PI / 2; i += step_size) {
        denominator += donelan_banner(omega, i) * swell_factor(omega, i);
    }

    return 1 / (denominator * step_size);
}

// Eqn 42
float directional_spreding(float omega, float theta) {
    return donelan_banner(omega, theta) * swell_factor(omega, theta) * normalization_factor(omega);
}

[numthreads(8, 8, 1)]
void create_initial_spectrum(uint2 thread_id: SV_DispatchThreadID) {
    // Wavenumber
    float delta_k = TWO_PI / ocean_params.L;
    float2 k = delta_k * (thread_id - float2(ocean_params.ocean_dim / 2, ocean_params.ocean_dim / 2));
    float k_length = length(k);

    if (k_length >= 0.0001f) {
        // Angular frequency
        float omega = omega_value(k_length);
        float theta = atan2(k.y, k.x) - ocean_params.wind_angle;

        float non_directional_spectrum = tma(omega);
        float directional_spectrum = directional_spreding(omega, theta);
        // Eqn 16
        float spectrum = non_directional_spectrum * directional_spectrum;

        float2 noise_value = bindless_textures[ocean_params.noise_tex_idx][thread_id];
        float2 h_0k = noise_value * sqrt(2 * spectrum * omega_derivative(k_length) / k_length * delta_k * delta_k);

        // Eqn 47
        bindless_rwtextures[ocean_params.waves_spectrum_idx][thread_id] = float4(h_0k.x, h_0k.y, k.x, k.y);
    } else {
        bindless_rwtextures[ocean_params.waves_spectrum_idx][thread_id] = 0.0;
    }
}