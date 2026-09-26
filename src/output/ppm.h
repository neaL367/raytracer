#pragma once
#include "../core/vec3.h"
#include "film.h"
#include <fstream>
#include <string>
#include <vector>

// Sole PPM writer: binary P6. HDR film stays linear vec3; exposure +
// ACES + sRGB applied at write, never in renderer. (P3 ASCII removed:
// identical pixels at ~3x the bytes, readers take P6.)
inline bool write_ppm(const char *path, const std::vector<vec3> &fb, int w, int h,
                      double exposure = 1.0) {
    std::string buf;
    buf.reserve((size_t)w * (size_t)h * 3 + 32);
    buf.append("P6\n");
    buf.append(std::to_string(w));
    buf.push_back(' ');
    buf.append(std::to_string(h));
    buf.append("\n255\n");
    for (int j = h - 1; j >= 0; --j) {
        for (int i = 0; i < w; ++i) {
            vec3 ldr = tonemap(fb[(size_t)j * w + i], exposure);
            buf.push_back(static_cast<char>(static_cast<int>(255.999 * ldr.x())));
            buf.push_back(static_cast<char>(static_cast<int>(255.999 * ldr.y())));
            buf.push_back(static_cast<char>(static_cast<int>(255.999 * ldr.z())));
        }
    }
    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    out.write(buf.data(), (std::streamsize)buf.size());
    return (bool)out;
}
