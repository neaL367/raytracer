#pragma once
#include "vec3.h"
#include <memory>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

    vec3 value(double u, double v, const vec3 &p) const override
    {
        return albedo;
    }

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

private:
    double inv_scale;
    std::shared_ptr<texture> even;
    std::shared_ptr<texture> odd;
};

// Image texture over a PPM file (P3 text or P6 binary, 8-bit). No third-party
// decoder: uncompressed PPM parses in ~40 lines and the project already
// speaks the format. Nearest sampling with clamped UVs; bilinear filtering
// is a later slice, not smuggled in here.
class image_texture : public texture
{
public:
    image_texture(const char *filename) { load_ppm(filename); }

    vec3 value(double u, double v, const vec3 &p) const override
    {
        if (data.empty())
            return vec3(1, 0, 1); // magenta = missing texture, never silent
        if (u < 0.0)
            u = 0.0;
        if (u > 1.0)
            u = 1.0;
        // Image row 0 is the top; sphere v = 1 is the north pole.
        double flipped = 1.0 - v;
        if (flipped < 0.0)
            flipped = 0.0;
        if (flipped > 1.0)
            flipped = 1.0;
        int x = std::min(width - 1, static_cast<int>(u * width));
        int y = std::min(height - 1, static_cast<int>(flipped * height));
        size_t i = static_cast<size_t>(y * width + x) * 3;
        const double s = 1.0 / 255.0;
        return vec3(data[i] * s, data[i + 1] * s, data[i + 2] * s);
    }

    bool needs_uv() const override { return true; }

private:
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
    std::vector<unsigned char> data;
};
