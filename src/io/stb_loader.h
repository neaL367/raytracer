#pragma once
// Any-format decode (JPEG/PNG/BMP/...) via stb_image, into linear HDR.
// sRGB sources decode with the exact inverse transfer (not gamma 2.2).
#include "core/vec3.h"
#include "io/ppm_image.h"
#include "stb/stb_image.h"

#include <cmath>
#include <string>

namespace stb_loader {

inline double srgb_to_linear(double v) {
    v = v < 0 ? 0 : (v > 1 ? 1 : v);
    return (v <= 0.04045) ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}

inline bool load_image(const std::string &path, ppm_io::image &img) {
    int w = 0, h = 0, ch = 0;
    unsigned char *px = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!px || w <= 0 || h <= 0)
        return false;
    img.w = w;
    img.h = h;
    img.px.resize((size_t)w * h);
    for (size_t i = 0; i < (size_t)w * h; ++i)
        img.px[i] = vec3(srgb_to_linear(px[3 * i] / 255.0),
                         srgb_to_linear(px[3 * i + 1] / 255.0),
                         srgb_to_linear(px[3 * i + 2] / 255.0));
    stbi_image_free(px);
    return true;
}

} // namespace stb_loader
