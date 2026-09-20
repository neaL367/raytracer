#pragma once
// Minimal PPM reader (P3 ascii + P6 binary) for image textures.
// Zero-dep by design; PNG via stb flagged later, not now.
#include "core/vec3.h"

#include <cctype>
#include <fstream>
#include <string>
#include <vector>

namespace ppm_io {

inline bool next_token(std::istream &in, std::string &tok) {
    tok.clear();
    char c;
    while (in.get(c)) {
        if (c == '#') {
            std::string skip;
            std::getline(in, skip);
            continue;
        }
        if (std::isspace((unsigned char)c)) {
            if (!tok.empty())
                return true;
            continue;
        }
        tok.push_back(c);
    }
    return !tok.empty();
}

struct image {
    int w = 0, h = 0;
    std::vector<vec3> px; // linear 0..1, row 0 = top
};

inline bool read_ppm(const std::string &path, image &img) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::string magic;
    if (!next_token(f, magic))
        return false;
    bool raw = (magic == "P6");
    if (!raw && magic != "P3")
        return false;
    std::string tok;
    if (!next_token(f, tok))
        return false;
    int w = std::stoi(tok);
    if (!next_token(f, tok))
        return false;
    int h = std::stoi(tok);
    if (!next_token(f, tok))
        return false;
    int maxv = std::stoi(tok);
    if (w <= 0 || h <= 0 || maxv <= 0 || maxv > 65535)
        return false;
    img.w = w;
    img.h = h;
    img.px.resize((size_t)w * h);
    if (raw) {
        // Single whitespace after maxval precedes raster.
        if (f.peek() == '\n' || f.peek() == '\r')
            f.get();
        if (maxv < 256) {
            std::vector<unsigned char> raw_px((size_t)w * h * 3);
            f.read((char *)raw_px.data(), (std::streamsize)raw_px.size());
            if ((size_t)f.gcount() != raw_px.size())
                return false;
            for (size_t i = 0; i < (size_t)w * h; ++i)
                img.px[i] = vec3(raw_px[3 * i] / (double)maxv, raw_px[3 * i + 1] / (double)maxv,
                                 raw_px[3 * i + 2] / (double)maxv);
        } else {
            std::vector<unsigned char> raw_px((size_t)w * h * 6);
            f.read((char *)raw_px.data(), (std::streamsize)raw_px.size());
            if ((size_t)f.gcount() != raw_px.size())
                return false;
            for (size_t i = 0; i < (size_t)w * h; ++i) {
                auto ch = [&](size_t k) {
                    return (raw_px[2 * k] * 256 + raw_px[2 * k + 1]) / (double)maxv;
                };
                size_t k = 3 * i;
                img.px[i] = vec3(ch(k), ch(k + 1), ch(k + 2));
            }
        }
    } else {
        for (size_t i = 0; i < (size_t)w * h; ++i) {
            double c[3];
            for (int k = 0; k < 3; ++k) {
                if (!next_token(f, tok))
                    return false;
                c[k] = std::stoi(tok) / (double)maxv;
            }
            img.px[i] = vec3(c[0], c[1], c[2]);
        }
    }
    return true;
}

} // namespace ppm_io
