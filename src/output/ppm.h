#pragma once
#include "../core/vec3.h"
#include "film.h"
#include <fstream>
#include <vector>

// PPM P3 writer. HDR film stays linear vec3; exposure + ACES + sRGB
// applied at write, never in renderer. Default exposure keeps old
// callers (GPU host) compiling; look shifts vs sqrt path, documented.
inline bool write_ppm(const char *path, const std::vector<vec3> &fb, int w, int h,
                      double exposure = 1.0) {
    std::ofstream out(path);
    if (!out)
        return false;
    out << "P3\n" << w << ' ' << h << "\n255\n";
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            vec3 ldr = tonemap(fb[(size_t)j * w + i], exposure);
            out << static_cast<int>(255.999 * ldr.x()) << ' '
                << static_cast<int>(255.999 * ldr.y()) << ' '
                << static_cast<int>(255.999 * ldr.z()) << '\n';
        }
    }
    return true;
}
