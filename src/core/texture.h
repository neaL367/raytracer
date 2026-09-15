#pragma once
#include "vec3.h"
#include <memory>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Minimal stb_image surface (v2.30, vendored under external/stb, implemented
// once in io/stb_image.cpp). Forward-declared here so third-party headers
// never enter every translation unit; signatures and C linkage must match
// stb_image.h (it wraps everything in extern "C").
extern "C" {
extern unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file,
                                int desired_channels);
extern void stbi_image_free(void *retval_from_stbi_load);
extern const char *stbi_failure_reason(void);
}

// Texture: color as a function of surface position. Materials sample this
// instead of holding a flat albedo, so one lambertian covers solids,
// checkers, and (next) image maps. u/v come from the hit record; p is the
// 3D hit point for procedural patterns that ignore parametrization.
class texture
{
public:
    virtual ~texture() = default;
    virtual vec3 value(double u, double v, const vec3 &p) const = 0;
    // True for parametrized maps (image textures); procedural patterns work
    // from the 3D point alone and cost no UV computation.
    virtual bool needs_uv() const { return false; }
};

class solid_color : public texture
{
public:
    solid_color(const vec3 &albedo) : albedo(albedo) {}

    vec3 value(double, double, const vec3 &) const override
    {
        return albedo;
    }

    // Read access for GPU upload.
    const vec3 &color() const { return albedo; }

private:
    vec3 albedo;
};

class checker_texture : public texture
{
public:
    checker_texture(double scale, std::shared_ptr<texture> even, std::shared_ptr<texture> odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

    checker_texture(double scale, const vec3 &c1, const vec3 &c2)
        : checker_texture(scale, std::make_shared<solid_color>(c1), std::make_shared<solid_color>(c2)) {}

    vec3 value(double u, double v, const vec3 &p) const override
    {
        // 3D checker: product of sines flips sign per cell along every axis,
        // so cells alternate in all three dimensions, not just on a plane.
        double sines = std::sin(inv_scale * p.x()) * std::sin(inv_scale * p.y()) * std::sin(inv_scale * p.z());
        return (sines < 0.0) ? odd->value(u, v, p) : even->value(u, v, p);
    }

    // Read access for GPU upload (even/odd are solid colors in practice).
    double scale() const { return 1.0 / inv_scale; }
    const std::shared_ptr<texture> &color_a() const { return even; }
    const std::shared_ptr<texture> &color_b() const { return odd; }

private:
    double inv_scale;
    std::shared_ptr<texture> even;
    std::shared_ptr<texture> odd;
};

// Image texture: PPM (P3/P6, in-house reader) or anything stb_image decodes
// (PNG/JPG/BMP/TGA...). Sampling is format-blind — pixels land in the same
// byte buffer either way, so a PNG and PPM of one picture render identically.
class image_texture : public texture
{
public:
    image_texture(const char *filename, bool bilinear = true) : bilinear(bilinear)
    {
        load_image(filename);
    }

    vec3 value(double u, double v, const vec3 &) const override
    {
        if (data.empty())
            return vec3(1, 0, 1); // magenta = missing texture, never silent
        // Image row 0 is the top; sphere v = 1 is the north pole.
        double x = clamp01(u) * width - 0.5;
        double y = clamp01(1.0 - v) * height - 0.5;
        if (!bilinear)
            return texel(static_cast<int>(x + 0.5), static_cast<int>(y + 0.5));
        // Bilinear: four surrounding texels weighted by fractional position.
        // The -0.5 centers the kernel so texel centers sit on integers.
        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));
        double fx = x - x0;
        double fy = y - y0;
        vec3 c00 = texel(x0, y0);
        vec3 c10 = texel(x0 + 1, y0);
        vec3 c01 = texel(x0, y0 + 1);
        vec3 c11 = texel(x0 + 1, y0 + 1);
        return c00 * ((1 - fx) * (1 - fy)) + c10 * (fx * (1 - fy)) +
               c01 * ((1 - fx) * fy) + c11 * (fx * fy);
    }

    bool needs_uv() const override { return true; }

    // Read access for GPU upload.
    int pixel_width() const { return width; }
    int pixel_height() const { return height; }
    const std::vector<unsigned char> &bytes() const { return data; }

private:
    static double clamp01(double t)
    {
        if (t < 0.0)
            return 0.0;
        if (t > 1.0)
            return 1.0;
        return t;
    }

    vec3 texel(int x, int y) const
    {
        if (x < 0)
            x = 0;
        if (x > width - 1)
            x = width - 1;
        if (y < 0)
            y = 0;
        if (y > height - 1)
            y = height - 1;
        size_t i = static_cast<size_t>(y * width + x) * 3;
        const double s = 1.0 / 255.0;
        return vec3(data[i] * s, data[i + 1] * s, data[i + 2] * s);
    }

    static bool has_ppm_suffix(const char *filename)
    {
        std::string name = filename;
        if (name.size() < 4)
            return false;
        std::string ext = name.substr(name.size() - 4);
        return ext == ".ppm" || ext == ".PPM";
    }

    void load_image(const char *filename)
    {
        if (has_ppm_suffix(filename))
        {
            load_ppm(filename);
            return;
        }
        int w = 0, h = 0, channels = 0;
        unsigned char *px = stbi_load(filename, &w, &h, &channels, 3);
        if (!px)
        {
            std::cerr << "image_texture: cannot decode " << filename
                      << " (" << stbi_failure_reason() << ")\n";
            return; // data stays empty -> magenta fallback
        }
        width = w;
        height = h;
        data.assign(px, px + static_cast<size_t>(w) * h * 3);
        stbi_image_free(px);
    }

    // Next whitespace-separated integer, skipping '#' comment lines.
    static bool next_int(std::istream &in, int &out)
    {
        while (true)
        {
            in >> std::ws;
            if (in.peek() == '#')
            {
                std::string ignored;
                std::getline(in, ignored);
                continue;
            }
            return static_cast<bool>(in >> out);
        }
    }

    void load_ppm(const char *filename)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
        {
            std::cerr << "image_texture: cannot open " << filename << "\n";
            return;
        }
        std::string magic;
        in >> magic;
        bool binary = (magic == "P6");
        if (magic != "P3" && !binary)
        {
            std::cerr << "image_texture: " << filename << " is not P3/P6 PPM\n";
            return;
        }
        int maxval = 0;
        if (!next_int(in, width) || !next_int(in, height) || !next_int(in, maxval) ||
            width <= 0 || height <= 0 || maxval != 255)
        {
            std::cerr << "image_texture: bad header in " << filename << "\n";
            width = height = 0;
            return;
        }
        data.resize(static_cast<size_t>(width) * height * 3);
        if (binary)
        {
            in.get(); // single whitespace byte after maxval
            in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!in)
            {
                std::cerr << "image_texture: truncated pixels in " << filename << "\n";
                data.clear();
            }
        }
        else
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                int v = 0;
                if (!(in >> v))
                {
                    std::cerr << "image_texture: truncated pixels in " << filename << "\n";
                    data.clear();
                    break;
                }
                data[i] = static_cast<unsigned char>(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    }

    int width = 0;
    int height = 0;
    bool bilinear = true;
    std::vector<unsigned char> data;
};
