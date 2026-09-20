#pragma once
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
};

class solid_color : public texture {
public:
    solid_color(const vec3 &c) : color(c) {}
    vec3 value(double, double, const vec3 &) const override { return color; }

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

private:
    double scale;
    std::shared_ptr<texture> even, odd;
};

class image_texture : public texture {
public:
    image_texture() {}
    image_texture(int w, int h, std::vector<vec3> px)
        : W(w), H(h), pixels(std::move(px)) {}
    vec3 value(double u, double v, const vec3 &) const override {
        if (pixels.empty())
            return vec3(0, 1, 1); // debug cyan: missing image
        if (u < 0)
            u = 0;
        if (u > 1)
            u = 1;
        if (v < 0)
            v = 0;
        if (v > 1)
            v = 1;
        // v=0 is bottom; image row 0 is top.
        int x = (int)(u * (W - 1) + 0.5);
        int y = (int)((1 - v) * (H - 1) + 0.5);
        return pixels[(size_t)y * W + x];
    }
    int width() const { return W; }
    int height() const { return H; }

private:
    int W = 0, H = 0;
    std::vector<vec3> pixels;
};
