#ifndef ENV_GLSL
#define ENV_GLSL

// Environment & HDRI Lighting Subsystem:
// - Analytic sun+sky radiance and direction sampling
// - Equirectangular HDRI radiance evaluation and 2D CDF importance sampling

// Analytic sun+sky (mirrors CPU env_light analytic paths).
vec3 env_sky(vec3 d) {
    vec3 unit = normalize(d);
    float tt = 0.5 * (unit.y + 1.0);
    return (1.0 - tt) * vec3(1.0) + tt * vec3(0.5, 0.7, 1.0);
}

vec3 env_sun_dir() {
    return normalize(vec3(0.5, 0.8, 0.35));
}

vec3 env_analytic_radiance(vec3 d) {
    vec3 ud = normalize(d);
    vec3 L = env_sky(ud);
    float s = dot(ud, env_sun_dir());
    if (s > 0.9993) {
        float k = (s - 0.9993) / (1.0 - 0.9993);
        L += vec3(30.0, 25.0, 18.0) * (k * k);
    }
    return L;
}

// ---- HDRI helpers (nenv == 2) ----------------------------------------
// UV convention matches CPU hdri_env and codebase sphere UVs:
//   u = (atan2(-z, x) + pi) / (2*pi),  v = acos(y) / pi

vec2 dir_to_hdri_uv(vec3 d) {
    vec3 ud = normalize(d);
    float phi = atan(-ud.z, ud.x);        // atan2(-z, x) ∈ [-pi, pi]
    float u   = (phi + 3.14159265) / 6.28318530;
    float v   = acos(clamp(ud.y, -1.0, 1.0)) / 3.14159265;
    return vec2(u, v);
}

vec3 hdri_uv_to_dir(float u, float v) {
    float theta = 3.14159265 * v;
    float phi   = 6.28318530 * u;
    float sinT  = sin(theta);
    float cosT  = cos(theta);
    // Inverse of dir_to_hdri_uv: x=-sinT*cos(phi), y=cosT, z=sinT*sin(phi)
    return normalize(vec3(-sinT * cos(phi), cosT, sinT * sin(phi)));
}

vec3 hdri_radiance(vec3 d) {
    int W = int(hdriCdf[0] + 0.5);
    int H = int(hdriCdf[1] + 0.5);
    if (W <= 0 || H <= 0) return vec3(0.0);
    vec2 uv = dir_to_hdri_uv(d);
    float u = clamp(uv.x, 0.0, 1.0);
    float v = clamp(uv.y, 0.0, 1.0);
    float fx = u * float(W - 1);
    float fy = (1.0 - v) * float(H - 1);  // v=0 → top row
    int x0 = int(floor(fx)), y0 = int(floor(fy));
    int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    float tx = fx - float(x0), ty = fy - float(y0);
    vec3 c00 = hdriTex[y0 * W + x0].xyz, c10 = hdriTex[y0 * W + x1].xyz;
    vec3 c01 = hdriTex[y1 * W + x0].xyz, c11 = hdriTex[y1 * W + x1].xyz;
    return c00 * ((1.0-tx)*(1.0-ty)) + c10*(tx*(1.0-ty)) +
           c01 * ((1.0-tx)*ty)       + c11*(tx*ty);
}

// Binary search: largest i in [0,n-1] where cdf[base+i+1] <= key.
int hdri_lb(int base, int n, float key) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (hdriCdf[base + mid + 1] <= key) lo = mid + 1;
        else hi = mid;
    }
    return clamp(lo, 0, n - 1);
}

// Sample from the HDRI CDF. Returns direction and sets pdf_out.
vec3 hdri_sample(float u1, float u2, out float pdf_out) {
    int W = int(hdriCdf[0] + 0.5);
    int H = int(hdriCdf[1] + 0.5);
    if (W <= 0 || H <= 0) { pdf_out = 1.0 / (4.0 * 3.14159265); return vec3(0,1,0); }
    int marg_base = 2;  // marginal CDF starts at hdriCdf[2]
    int vi = hdri_lb(marg_base, H, u1);
    float v0 = hdriCdf[marg_base + vi], v1 = hdriCdf[marg_base + vi + 1];
    float fv  = (v1 > v0) ? (u1 - v0) / (v1 - v0) : 0.5;
    float vf  = (float(vi) + fv) / float(H);

    int cond_base = 2 + H + 1 + vi * (W + 1);
    int ui = hdri_lb(cond_base, W, u2);
    float c0 = hdriCdf[cond_base + ui], c1 = hdriCdf[cond_base + ui + 1];
    float fu  = (c1 > c0) ? (u2 - c0) / (c1 - c0) : 0.5;
    float uf  = (float(ui) + fu) / float(W);

    vec3 dir = hdri_uv_to_dir(uf, vf);
    float sinT = length(vec2(dir.x, dir.z));  // sin(theta) = sqrt(1-y^2)
    float p_uv = (v1 - v0) * float(H) * (c1 - c0) * float(W);
    pdf_out = (sinT > 1e-4) ? p_uv / (2.0 * 3.14159265 * 3.14159265 * sinT) : 0.0;
    return dir;
}

// Evaluate HDRI PDF for a given direction (for MIS miss weight).
float hdri_pdf(vec3 d) {
    int W = int(hdriCdf[0] + 0.5);
    int H = int(hdriCdf[1] + 0.5);
    if (W <= 0 || H <= 0) return 1.0 / (4.0 * 3.14159265);
    vec2 uv = dir_to_hdri_uv(d);
    float sinT = sin(3.14159265 * uv.y);
    if (sinT < 1e-4) return 0.0;
    int vi = clamp(int(uv.y * float(H)), 0, H - 1);
    float marg_diff = (hdriCdf[2 + vi + 1] - hdriCdf[2 + vi]) * float(H);
    int ui = clamp(int(uv.x * float(W)), 0, W - 1);
    int cb = 2 + H + 1 + vi * (W + 1);
    float cond_diff = (hdriCdf[cb + ui + 1] - hdriCdf[cb + ui]) * float(W);
    return marg_diff * cond_diff / (2.0 * 3.14159265 * 3.14159265 * sinT);
}

// Analytic uniform-sphere sample (parity with old env_sample_dir).
vec3 env_analytic_sample_dir(float u1, float u2) {
    float z   = 1.0 - 2.0 * u1;
    float phi = 6.28318530 * u2;
    float r   = sqrt(max(1.0 - z * z, 0.0));
    return vec3(r * cos(phi), r * sin(phi), z);
}

// ---- Dispatch wrappers (used by Li) ------------------------------------
vec3 env_radiance(vec3 d) {
    if (pc.nenv == 2) return hdri_radiance(d);
    return env_analytic_radiance(d);
}

vec3 env_sample(float u1, float u2, out float pdf) {
    if (pc.nenv == 2) return hdri_sample(u1, u2, pdf);
    pdf = 1.0 / (4.0 * 3.14159265);
    return env_analytic_sample_dir(u1, u2);
}

float env_pdf_for(vec3 d) {
    if (pc.nenv == 2) return hdri_pdf(d);
    return 1.0 / (4.0 * 3.14159265);
}

#endif // ENV_GLSL
