#pragma once
// PPM output: tonemaps the linear framebuffer, quantizes to bytes, and
// optionally hashes the exact bytes written (FNV-1a, one pass with the
// write loop). Returns the hash, or 0 when hashing is off.
#include "../core/vec3.h"
#include "tonemap.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

inline std::uint64_t write_ppm(const std::string &path, const std::vector<vec3> &framebuffer,
                               int image_width, int image_height, double exposure, bool do_hash)
{
    std::uint64_t image_hash = 1469598103934665603ULL;
    auto hash_byte = [&](unsigned char b)
    {
        image_hash ^= b;
        image_hash *= 1099511628211ULL;
    };

    std::ofstream out(path);
    out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            vec3 c = tonemap(framebuffer[j * image_width + i], exposure);
            // Final guard: tonemapped values sit in [0,1] by construction;
            // the clamp only catches the fit's asymptote overshoot past 1.
            int ir = std::min(255, static_cast<int>(255.999 * c.x()));
            int ig = std::min(255, static_cast<int>(255.999 * c.y()));
            int ib = std::min(255, static_cast<int>(255.999 * c.z()));
            out << ir << ' ' << ig << ' ' << ib << '\n';
            if (do_hash)
            {
                hash_byte(static_cast<unsigned char>(ir));
                hash_byte(static_cast<unsigned char>(ig));
                hash_byte(static_cast<unsigned char>(ib));
            }
        }
    }

    return do_hash ? image_hash : 0;
}
