#pragma once
#include "../core/vec3.h"
#include <cstdint>
#include <fstream>
#include <vector>

// PFM color float dump: linear pre-exposure film for post-workflows.
// Header "PF", dims, "-1.0" (little-endian), raw RGB f32 rows bottom-first
// (fb j=0 is bottom, mirroring write_ppm's top-first text order).
inline bool write_pfm(const char *path, const std::vector<vec3> &fb, int w, int h) {
    if ((int)fb.size() != w * h || w <= 0 || h <= 0)
        return false;
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    out << "PF\n" << w << ' ' << h << "\n-1.0\n";
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i) {
            const vec3 &p = fb[(size_t)j * w + i];
            float rgb[3] = {(float)p.x(), (float)p.y(), (float)p.z()};
            static_assert(sizeof(float) == 4);
            out.write((const char *)rgb, sizeof rgb);
        }
    return (bool)out;
}
