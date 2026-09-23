#ifndef NOISE_GLSL
#define NOISE_GLSL

// Procedural Value Noise and 3D Marble Turbulence:
// Mirrors CPU value_noise and noise_texture.

uint nlattice(ivec3 c) {
    uint h = uint(c.x) * 374761393u + uint(c.y) * 668265263u + uint(c.z) * 1440662683u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

float nunit(ivec3 c) {
    return float(nlattice(c) & 0xffffu) / 65535.0;
}

float vnoise(vec3 p) {
    ivec3 b = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float c000 = nunit(b), c100 = nunit(b + ivec3(1, 0, 0));
    float c010 = nunit(b + ivec3(0, 1, 0)), c110 = nunit(b + ivec3(1, 1, 0));
    float c001 = nunit(b + ivec3(0, 0, 1)), c101 = nunit(b + ivec3(1, 0, 1));
    float c011 = nunit(b + ivec3(0, 1, 1)), c111 = nunit(b + ivec3(1, 1, 1));
    float x00 = mix(c000, c100, u.x), x10 = mix(c010, c110, u.x);
    float x01 = mix(c001, c101, u.x), x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

float vturb(vec3 p, int depth) {
    float sum = 0.0, weight = 0.0, amp = 1.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= depth)
            break;
        sum += amp * abs(vnoise(p));
        weight += amp;
        amp *= 0.5;
        p *= 2.0;
    }
    return weight > 0.0 ? sum / weight : 0.0;
}

// Mode 0 = raw, 1 = turbulence, 2 = marble. Matches CPU noise_texture.
vec3 noise_col(vec3 p, vec3 c0, vec3 c1, float freq, int depth, int mode) {
    vec3 q = p * freq;
    float f = vnoise(q);
    if (mode == 1)
        f = vturb(q, depth);
    else if (mode == 2)
        f = 0.5 * (1.0 + sin(freq * p.z + 10.0 * vturb(q, depth)));
    return c0 * (1.0 - f) + c1 * f;
}

#endif // NOISE_GLSL
