#ifndef SHADING_GLSL
#define SHADING_GLSL

// Shading & Optics Subsystem for GPU Path Tracer:
// - GGX Microfacet (VNDF, isotropic & anisotropic Smith shadowing)
// - Complex numbers arithmetic (cmul, cdiv, cabs2, csqrt)
// - Exact Conductor Fresnel (conductor_R, nk_preset)
// - Airy Thin-Film Interference (film_R, film_R_rgb)
// - Hero-Wavelength Spectral Transport (spick, hero_lambda, cauchy_ior)

// GGX conductor core: mirrors CPU ggx.h (fp32). Height-correlated
// Smith ratio + Schlick F0 + Heitz VNDF, 2 RNG draws like CPU.
float ggx_lambda(float alpha, float cos_w) {
    float t = sqrt(max(1.0 - cos_w * cos_w, 0.0)) / cos_w;
    float a = alpha * t;
    return (sqrt(1.0 + a * a) - 1.0) * 0.5;
}

vec3 checker_col(vec3 p, vec3 even, vec3 odd, float scale) {
    int s = int(floor(scale * p.x)) + int(floor(scale * p.y)) + int(floor(scale * p.z));
    return (s % 2 == 0) ? even : odd;
}

vec3 ggx_vndf(float alpha, vec3 V, float u1, float u2, out vec3 H) {
    vec3 Vh = normalize(vec3(alpha * V.x, alpha * V.y, V.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(1.0 - t1 * t1, 0.0)) + s * t2;
    vec3 Nh = T1 * t1 + T2 * t2 + Vh * sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0));
    H = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(Nh.z, 0.0)));
    return H * (2.0 * dot(V, H)) - V; // reflect incident (-V) about H
}

// Anisotropic GGX core: mirrors CPU ggx aniso fns (fp32). Height-
// correlated Smith ratio + Heitz anisotropic VNDF, 2 RNG draws like CPU.
float ggx_lambda_aniso(float ax, float ay, vec3 w) {
    if (w.z <= 0.0)
        return 1e30;
    float tx = w.x / w.z;
    float ty = w.y / w.z;
    float a2 = (ax * tx) * (ax * tx) + (ay * ty) * (ay * ty);
    return (sqrt(1.0 + a2) - 1.0) * 0.5;
}

vec3 ggx_vndf_aniso(float ax, float ay, vec3 V, float u1, float u2, out vec3 H) {
    vec3 Vh = normalize(vec3(ax * V.x, ay * V.y, V.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(1.0 - t1 * t1, 0.0)) + s * t2;
    vec3 Nh = T1 * t1 + T2 * t2 + Vh * sqrt(max(1.0 - t1 * t1 - t2 * t2, 0.0));
    H = normalize(vec3(ax * Nh.x, ay * Nh.y, max(Nh.z, 0.0)));
    return H * (2.0 * dot(V, H)) - V;
}

// Power heuristic, one sample each: mirrors CPU direction_pdf::power_weight.
float power_weight(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a, b2 = pdf_b * pdf_b;
    float denom = a2 + b2;
    return denom > 0.0 ? a2 / denom : 0.0;
}

// Push word mixpdf carries two flags (M63 nenv precedent): bit 0 =
// Book-3 mixture, bit 1 = M67 hero-wavelength spectral transport.
bool use_mix() { return (pc.mixpdf & 1) != 0; }
bool use_spectral() { return (pc.mixpdf & 2) != 0; }

// Channel-pick (mirrors spectrum::pick): identity when spectral is off,
// else the hero channel value. The 1/3 hero probability is compensated
// once, by scaling the finished path x3 at the end of Li().
vec3 spick(vec3 v, int hero) {
    if (!use_spectral() || hero < 0)
        return v;
    vec3 z = vec3(0.0);
    if (hero == 0)
        z.x = v.x;
    else if (hero == 1)
        z.y = v.y;
    else
        z.z = v.z;
    return z;
}

float hero_lambda(int hero) {
    if (hero == 0)
        return 650.0;
    return (hero == 1) ? 550.0 : 450.0;
}

float cauchy_ior(float nref, float B, float lambda_nm) {
    if (B == 0.0)
        return nref;
    float l = lambda_nm / 1000.0;
    float lr = 0.5876;
    return nref + B * (1.0 / (l * l) - 1.0 / (lr * lr));
}

// ---- vec2 complex core (mirrors spectrum.h cplx) ----
vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 cdiv(vec2 a, vec2 b) {
    float d = dot(b, b);
    return vec2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

float cabs2(vec2 a) { return dot(a, a); }

vec2 csqrt(vec2 a) {
    float r = length(a);
    float t = 0.5 * atan(a.y, a.x);
    float s = sqrt(r);
    return vec2(s * cos(t), s * sin(t));
}

// Exact unpolarized conductor Fresnel (mirrors spectrum::conductor_R).
float conductor_R(float n0, float n, float k, float cos_ti) {
    float c = clamp(cos_ti, 0.0, 1.0);
    float s2 = 1.0 - c * c;
    vec2 N = vec2(n, k);
    vec2 cos2t = vec2(1.0, 0.0) - (n0 * n0 * s2) * cdiv(vec2(1.0, 0.0), cmul(N, N));
    vec2 cost = csqrt(cos2t);
    vec2 rs = cdiv(vec2(n0 * c, 0.0) - cmul(N, cost), vec2(n0 * c, 0.0) + cmul(N, cost));
    vec2 rp = cdiv(cmul(N, vec2(c, 0.0)) - vec2(n0, 0.0) * cost,
                   cmul(N, vec2(c, 0.0)) + vec2(n0, 0.0) * cost);
    return 0.5 * (cabs2(rs) + cabs2(rp));
}

// Amplitude Fresnel pair for one interface (mirrors spectrum::fresnel_ri).
void fresnel_ri_glsl(float n0re, float n0im, float n1re, float n1im, float cos_ti,
                     out vec2 rs, out vec2 rp) {
    float c = clamp(cos_ti, 0.0, 1.0);
    float s2 = 1.0 - c * c;
    vec2 N0 = vec2(n0re, n0im), N1 = vec2(n1re, n1im);
    vec2 cos2t = vec2(1.0, 0.0) - s2 * cdiv(cmul(N0, N0), cmul(N1, N1));
    vec2 cost = csqrt(cos2t);
    rs = cdiv(N0 * c - cmul(N1, cost), N0 * c + cmul(N1, cost));
    rp = cdiv(cmul(N1, vec2(c, 0.0)) - N0 * cost, cmul(N1, vec2(c, 0.0)) + N0 * cost);
}

// Thin-film Airy reflectance, one wavelength (mirrors thinfilm::film_R).
float film_R(float n0, float nf, float d_nm, float ns_n, float ns_k,
             float cos_ti, float lambda_nm) {
    float c = clamp(cos_ti, 0.0, 1.0);
    if (d_nm <= 0.0) {
        if (ns_k == 0.0) {
            vec2 rs, rp;
            fresnel_ri_glsl(n0, 0.0, ns_n, 0.0, c, rs, rp);
            return 0.5 * (cabs2(rs) + cabs2(rp));
        }
        return conductor_R(n0, ns_n, ns_k, c);
    }
    float s2 = 1.0 - c * c;
    float sin2t1 = (n0 * n0 / (nf * nf)) * s2;
    if (sin2t1 >= 1.0)
        return 1.0;
    float cost1 = sqrt(1.0 - sin2t1);
    float rs12 = (n0 * c - nf * cost1) / (n0 * c + nf * cost1);
    float rp12 = (nf * c - n0 * cost1) / (nf * c + n0 * cost1);
    vec2 rs23, rp23;
    fresnel_ri_glsl(nf, 0.0, ns_n, ns_k, cost1, rs23, rp23);
    float delta = 2.0 * 3.14159265 * nf * d_nm * cost1 / lambda_nm;
    vec2 e = vec2(cos(2.0 * delta), sin(2.0 * delta));
    vec2 nums = vec2(rs12, 0.0) + cmul(rs23, e);
    vec2 dens = vec2(1.0, 0.0) + rs12 * cmul(rs23, e);
    vec2 nump = vec2(rp12, 0.0) + cmul(rp23, e);
    vec2 denp = vec2(1.0, 0.0) + rp12 * cmul(rp23, e);
    return 0.5 * (cabs2(cdiv(nums, dens)) + cabs2(cdiv(nump, denp)));
}

vec3 film_R_rgb(float n0, float nf, float d_nm, float ns_n, float ns_k, float cos_ti) {
    return vec3(film_R(n0, nf, d_nm, ns_n, ns_k, cos_ti, 650.0),
                film_R(n0, nf, d_nm, ns_n, ns_k, cos_ti, 550.0),
                film_R(n0, nf, d_nm, ns_n, ns_k, cos_ti, 450.0));
}

bool nk_preset(int id, int ch, out float n, out float k) {
    if (ch < 0 || ch > 2)
        return false;
    if (id == 1) {
        n = (ch == 0) ? 0.17 : ((ch == 1) ? 0.35 : 1.45);
        k = (ch == 0) ? 3.80 : ((ch == 1) ? 2.75 : 1.95);
    } else if (id == 2) {
        n = (ch == 0) ? 0.07 : ((ch == 1) ? 0.13 : 0.16);
        k = (ch == 0) ? 4.20 : ((ch == 1) ? 4.00 : 2.60);
    } else if (id == 3) {
        n = (ch == 0) ? 0.28 : ((ch == 1) ? 0.95 : 1.25);
        k = (ch == 0) ? 3.40 : ((ch == 1) ? 2.65 : 2.40);
    } else if (id == 4) {
        n = (ch == 0) ? 1.45 : ((ch == 1) ? 0.95 : 0.65);
        k = (ch == 0) ? 7.60 : ((ch == 1) ? 6.00 : 5.30);
    } else {
        return false;
    }
    return true;
}

#endif // SHADING_GLSL
