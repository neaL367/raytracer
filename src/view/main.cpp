// rt_view: pixel-exact PPM preview + inspector + diff mode.
// Usage: rt_view [image.ppm] [--diff other.ppm] [--scale N] [--stats]
// Keys: mouse = x/y/RGB readout, D = diff heat, R = reload, Esc = quit.
// File watcher reloads on change (render-while-watch). --stats prints
// numbers without opening a window (headless/test safe).
#include "io/compare.h"
#include "io/ppm_image.h"

#include <SDL3/SDL.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> to_u8(const ppm_io::image &img) {
    std::vector<uint8_t> out;
    out.reserve((size_t)img.w * img.h * 3);
    for (const vec3 &p : img.px) {
        auto q = [](double v) {
            v = v < 0 ? 0 : (v > 1 ? 1 : v);
            return (uint8_t)(v * 255.999);
        };
        out.push_back(q(p.x()));
        out.push_back(q(p.y()));
        out.push_back(q(p.z()));
    }
    return out;
}

void print_stats(const char *label, const std::vector<uint8_t> &px, int w, int h) {
    double mean = 0;
    for (auto b : px)
        mean += b;
    mean /= px.size();
    std::cout << label << " " << w << "x" << h << " mean=" << mean << "\n";
}

} // namespace

int main(int argc, char **argv) {
    std::string path = "out/image.ppm";
    std::string diff_path;
    int scale = 2;
    bool stats_only = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--diff" && i + 1 < argc)
            diff_path = argv[++i];
        else if (a == "--scale" && i + 1 < argc)
            scale = std::max(1, std::atoi(argv[++i]));
        else if (a == "--stats")
            stats_only = true;
        else if (a.ends_with(".ppm"))
            path = a;
    }

    ppm_io::image img;
    if (!ppm_io::read_ppm(path, img)) {
        std::cerr << "cannot read " << path << " (P3/P6 only)\n";
        return 1;
    }
    std::vector<uint8_t> px = to_u8(img);
    std::vector<uint8_t> diff_px;
    bool show_diff = false;
    if (!diff_path.empty()) {
        ppm_io::image img2;
        if (!ppm_io::read_ppm(diff_path, img2) || img2.w != img.w || img2.h != img.h) {
            std::cerr << "diff image missing or size mismatch\n";
            return 1;
        }
        diff_px = to_u8(img2);
        diff_stats st = compare_images(px, diff_px, img.w, img.h);
        std::cout << "diff mean=" << st.mean_abs << " max=" << st.max_abs
                  << " over8=" << (st.frac_over * 100) << "%\n";
    }
    if (stats_only) {
        print_stats("image", px, img.w, img.h);
        return 0;
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }
    SDL_Window *win = SDL_CreateWindow(path.c_str(), img.w * scale, img.h * scale, 0);
    if (!win) {
        std::cerr << "window: " << SDL_GetError() << "\n";
        return 1;
    }
    SDL_Renderer *ren = SDL_CreateRenderer(win, nullptr);
    SDL_Texture *tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB24,
                                         SDL_TEXTUREACCESS_STREAMING, img.w, img.h);
    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_NEAREST); // pixel-exact
    auto upload = [&] {
        const std::vector<uint8_t> *src = &px;
        std::vector<uint8_t> heat;
        if (show_diff && !diff_px.empty()) {
            heat = diff_heatmap(px, diff_px, img.w, img.h);
            src = &heat;
        }
        SDL_UpdateTexture(tex, nullptr, src->data(), img.w * 3);
    };
    upload();

    auto mtime = std::filesystem::last_write_time(path);
    auto last_poll = std::chrono::steady_clock::now();
    bool running = true;
    char title[256];
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT)
                running = false;
            else if (e.type == SDL_EVENT_KEY_DOWN) {
                if (e.key.key == SDLK_ESCAPE)
                    running = false;
                else if (e.key.key == SDLK_D && !diff_px.empty()) {
                    show_diff = !show_diff;
                    upload();
                } else if (e.key.key == SDLK_R) {
                    if (ppm_io::read_ppm(path, img)) {
                        px = to_u8(img);
                        upload();
                    }
                }
            } else if (e.type == SDL_EVENT_MOUSE_MOTION) {
                int ix = (int)(e.motion.x / scale), iy = (int)(e.motion.y / scale);
                if (ix >= 0 && ix < img.w && iy >= 0 && iy < img.h) {
                    const uint8_t *p = &px[((size_t)iy * img.w + ix) * 3];
                    std::snprintf(title, sizeof title, "%s  x=%d y=%d  R=%d G=%d B=%d",
                                  path.c_str(), ix, iy, p[0], p[1], p[2]);
                    SDL_SetWindowTitle(win, title);
                }
            }
        }
        // Watcher: reload on external change (render-while-watch).
        auto now = std::chrono::steady_clock::now();
        if (now - last_poll > std::chrono::milliseconds(500)) {
            last_poll = now;
            auto mt = std::filesystem::last_write_time(path);
            if (mt != mtime) {
                mtime = mt;
                if (ppm_io::read_ppm(path, img)) {
                    px = to_u8(img);
                    upload();
                }
            }
        }
        SDL_RenderClear(ren);
        SDL_RenderTexture(ren, tex, nullptr, nullptr);
        SDL_RenderPresent(ren);
        SDL_Delay(16);
    }
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
