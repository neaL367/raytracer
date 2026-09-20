#pragma once
#include "../core/vec3.h"
#include <fstream>
#include <vector>

// PPM P3 writer. Reason: zero-dep, human-readable, Windows-trivial.
// HDR film stays linear vec3; gamma applied at write, not in renderer.
inline bool write_ppm(const char *path, const std::vector<vec3> &fb, int w, int h) {
    std::ofstream out(path);
    if (!out)
        return false;
    out << "P3\n" << w << ' ' << h << "\n255\n";
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            const vec3 &px = fb[j * w + i];
            // gamma 2.0 approx via sqrt, clamp to [0,1)
            double r = std::sqrt(px.x() < 0 ? 0 : (px.x() > 1 ? 1 : px.x()));
            double g = std::sqrt(px.y() < 0 ? 0 : (px.y() > 1 ? 1 : px.y()));
            double b = std::sqrt(px.z() < 0 ? 0 : (px.z() > 1 ? 1 : px.z()));
            out << static_cast<int>(255.999 * r) << ' '
                << static_cast<int>(255.999 * g) << ' '
                << static_cast<int>(255.999 * b) << '\n';
        }
    }
    return true;
}
