#pragma once
#include "noise.h"
#include "vec3.h"
#include <cmath>
#include <memory>
#include <vector>

// Texture: color as f(uv, point). Materials sample it; geometry owns uvs.
// Solid keeps old vec3 behavior; checker/image add visible mapping.
class texture {
public:
    virtual ~texture() = default;
    virtual vec3 value(double u, double v, const vec3 &p) const = 0;
    // Distance-aware sample: base ignores t (level 0). Image overrides.
    virtual vec3 sample(double u, double v, const vec3 &p, double t) const {
        (void)t;
        return value(u, v, p);
    }
};

// Render-height seam for LOD: app sets once per render (mirrors GPU pc.H).
inline int &mip_render_h() {
    static thread_local int h = 225;
    return h;
}
// Distance LOD: one level per doubling past the distance where a full
// texture spans `span` world units on screen. span is per-texture
// (ground sphere vs cube face differ 1000x); identical formula mirrored
// in path.comp.
inline double mip_select(double t, int iw, int ih, int render_h, double span) {
    double tex = (double)(iw > ih ? iw : ih) / ((double)render_h * span);
    double lod = std::log2(std::max(t, 1e-3) * tex);
    return lod < 0 ? 0 : lod;
}

class solid_color : public texture {
public:
    solid_color(const vec3 &c) : color(c) {}
    vec3 value(double, double, const vec3 &) const override { return color; }
    const vec3 &rgb() const { return color; }

private:
    vec3 color;
};

class checker : public texture {
public:
    checker(double s, std::shared_ptr<texture> e, std::shared_ptr<texture> o)
        : scale(s), even(e), odd(o) {}
    checker(double s, const vec3 &e, const vec3 &o)
        : scale(s), even(std::make_shared<solid_color>(e)),
          odd(std::make_shared<solid_color>(o)) {}
    vec3 value(double u, double v, const vec3 &p) const override {
        // World-pos parity: works on any shape, no UV dependence.
        // Negative coords: C++ % keeps sign, still != 0 -> odd branch.
        int s = (int)std::floor(scale * p.x()) + (int)std::floor(scale * p.y()) +
                (int)std::floor(scale * p.z());
        return (s % 2 == 0) ? even->value(u, v, p) : odd->value(u, v, p);
    }
    // GPU export accessors (flatten reads the pattern, not the code).
    double tex_scale() const { return scale; }
    const std::shared_ptr<texture> &tex_even() const { return even; }
    const std::shared_ptr<texture> &tex_odd() const { return odd; }

private:
    double scale;
    std::shared_ptr<texture> even, odd;
};

class image_texture : public texture {
public:
    struct mip_level {
        int w = 0, h = 0;
        std::vector<vec3> px;
    };
    image_texture() {}
    image_texture(int w, int h, std::vector<vec3> px, double span = 8.0)
        : W(w), H(h), pixels(std::move(px)), world_span(span) {
        build_chain();
    }
    vec3 value(double u, double v, const vec3 &) const override {
        if (pixels.empty())
            return vec3(0, 1, 1); // debug cyan: missing image
        return bilinear(chain[0], u, v);
    }
    // Trilinear over the pyramid at distance-selected LOD. lod 0 falls
    // through to the exact legacy bilinear path (bit-exact when fr == 0).
    vec3 sample(double u, double v, const vec3 &p, double t) const override {
        if (pixels.empty())
            return value(u, v, p);
        double lod = mip_select(t, W, H, mip_render_h(), world_span);
        if (lod > (double)(chain.size() - 1))
            lod = (double)(chain.size() - 1);
        int l0 = (int)lod;
        double fr = lod - l0;
        vec3 a = bilinear(chain[(size_t)l0], u, v);
        if (fr <= 0 || (size_t)l0 + 1 >= chain.size())
            return a;
        vec3 b = bilinear(chain[(size_t)l0 + 1], u, v);
        return a * (1 - fr) + b * fr;
    }
    int width() const { return W; }
    int height() const { return H; }
    double span() const { return world_span; }
    // GPU upload accessor (linear HDR texels, row 0 = top).
    const std::vector<vec3> &texels() const { return pixels; }
    const std::vector<mip_level> &mip_chain() const { return chain; }

private:
    static vec3 bilinear(const mip_level &m, double u, double v) {
        if (u < 0)
            u = 0;
        if (u > 1)
            u = 1;
        if (v < 0)
            v = 0;
        if (v > 1)
            v = 1;
        // Bilinear, v=0 bottom vs row 0 top. Edges clamp (x1/y1 pinned).
        double fx = u * (m.w - 1), fy = (1 - v) * (m.h - 1);
        int x0 = (int)fx, y0 = (int)fy;
        int x1 = x0 + 1 < m.w ? x0 + 1 : x0;
        int y1 = y0 + 1 < m.h ? y0 + 1 : y0;
        double tx = fx - x0, ty = fy - y0;
        const vec3 &c00 = m.px[(size_t)y0 * m.w + x0];
        const vec3 &c10 = m.px[(size_t)y0 * m.w + x1];
        const vec3 &c01 = m.px[(size_t)y1 * m.w + x0];
        const vec3 &c11 = m.px[(size_t)y1 * m.w + x1];
        return c00 * ((1 - tx) * (1 - ty)) + c10 * (tx * (1 - ty)) +
               c01 * ((1 - tx) * ty) + c11 * (tx * ty);
    }
    void build_chain() {
        chain.clear();
        if (pixels.empty() || W <= 0 || H <= 0)
            return;
        chain.push_back({W, H, pixels});
        // Box-filter pyramid, odd dims clamp (average actual coverage).
        while (chain.back().w > 1 || chain.back().h > 1) {
            const mip_level &up = chain.back();
            int nw = up.w > 1 ? up.w / 2 : 1;
            int nh = up.h > 1 ? up.h / 2 : 1;
            mip_level dn{nw, nh, std::vector<vec3>((size_t)nw * nh)};
            for (int y = 0; y < nh; ++y)
                for (int x = 0; x < nw; ++x) {
                    int x0 = x * 2, y0 = y * 2;
                    int x1 = x0 + 1 < up.w ? x0 + 1 : x0;
                    int y1 = y0 + 1 < up.h ? y0 + 1 : y0;
                    vec3 s = up.px[(size_t)y0 * up.w + x0] +
                             up.px[(size_t)y0 * up.w + x1] +
                             up.px[(size_t)y1 * up.w + x0] +
                             up.px[(size_t)y1 * up.w + x1];
                    int n = (x1 == x0 ? 1 : 2) * (y1 == y0 ? 1 : 2);
                    dn.px[(size_t)y * nw + x] = s * (1.0 / n);
                }
            chain.push_back(std::move(dn));
        }
    }

    int W = 0, H = 0;
    double world_span = 8.0; // world units one full texture spans
    std::vector<vec3> pixels;
    std::vector<mip_level> chain;
};

// Procedural marble family (NTW Ch.5 flavor): raw smoothed noise, fBm
// turbulence, or sine-banded marble over turbulence. World-pos like checker,
// so any shape maps without UVs. Mode 0 = raw, 1 = turb, 2 = marble.
class noise_texture : public texture {
public:
    noise_texture(double freq, int depth, int mode, const vec3 &c0, const vec3 &c1)
        : scale(freq), octaves(depth < 1 ? 1 : (depth > 8 ? 8 : depth)),
          kind(mode < 0 ? 0 : (mode > 2 ? 2 : mode)), col0(c0), col1(c1) {}
    vec3 value(double, double, const vec3 &p) const override {
        vec3 q = p * scale;
        double f = value_noise::at(q);
        if (kind == 1)
            f = value_noise::turb(q, octaves);
        else if (kind == 2)
            f = 0.5 * (1.0 + std::sin(scale * p.z() + 10.0 * value_noise::turb(q, octaves)));
        return col0 * (1.0 - f) + col1 * f;
    }
    // GPU export accessors (type-9 params, not code).
    double freq() const { return scale; }
    int depth() const { return octaves; }
    int mode() const { return kind; }
    const vec3 &color0() const { return col0; }
    const vec3 &color1() const { return col1; }

private:
    double scale;
    int octaves, kind;
    vec3 col0, col1;
};
