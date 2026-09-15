// Watcher window (V1): displays a project PPM, reloads it when the file
// changes (mtime poll). Workflow: render in one terminal, watch here.
// Keys: R reload now, Esc/Q quit. --frames N exits after N frames (headless
// smoke tests via SDL_VIDEODRIVER=dummy). Usage: rt_view [image.ppm]
// (default out/output.ppm).
#ifdef _MSC_VER
#pragma warning(push, 0) // third-party headers stay warning-free
#endif
#include <SDL3/SDL.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "../core/texture.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

struct LoadedImage
{
    int w = 0;
    int h = 0;
    std::vector<unsigned char> rgb; // tightly packed RGB888
};

bool load_ppm_rgb(const std::string &path, LoadedImage &img)
{
    image_texture tex(path.c_str());
    if (tex.bytes().empty())
        return false;
    img.w = tex.pixel_width();
    img.h = tex.pixel_height();
    img.rgb = tex.bytes();
    return true;
}

std::filesystem::file_time_type mtime_of(const std::string &path)
{
    std::error_code ec;
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec)
        return std::filesystem::file_time_type::min();
    return t;
}

} // namespace

int main(int argc, char **argv)
{
    std::string path = (argc > 1 && std::string(argv[1]) != "--frames") ? argv[1] : "out/output.ppm";
    int max_frames = 0;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--frames")
            max_frames = std::atoi(argv[i + 1]);
    // Allow --frames before the path too.
    if (argc > 1 && std::string(argv[1]) == "--frames" && argc > 3)
        path = argv[3];

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::fprintf(stderr, "viewer: SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    LoadedImage img;
    if (!load_ppm_rgb(path, img))
    {
        std::fprintf(stderr, "viewer: cannot load %s (render first?)\n", path.c_str());
        SDL_Quit();
        return 1;
    }
    std::printf("viewer: %s %dx%d\n", path.c_str(), img.w, img.h);

    SDL_Window *window = SDL_CreateWindow(path.c_str(), img.w, img.h, SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        std::fprintf(stderr, "viewer: window failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer)
    {
        std::fprintf(stderr, "viewer: renderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                                             SDL_TEXTUREACCESS_STREAMING, img.w, img.h);
    if (!texture)
    {
        std::fprintf(stderr, "viewer: texture failed: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_UpdateTexture(texture, nullptr, img.rgb.data(), img.w * 3);

    auto last_write = mtime_of(path);
    Uint64 last_poll = 0;
    int frames = 0;
    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
                running = false;
            else if (event.type == SDL_EVENT_KEY_DOWN)
            {
                if (event.key.key == SDLK_ESCAPE || event.key.key == SDLK_Q)
                    running = false;
                else if (event.key.key == SDLK_R)
                    last_write = std::filesystem::file_time_type::min(); // force reload below
            }
        }
        // mtime poll at ~2Hz: cheap, no platform file-watch API needed.
        // (A missed reload between polls just arrives 500ms later.)
        Uint64 now = SDL_GetTicks();
        if (now - last_poll > 500)
        {
            last_poll = now;
            auto current = mtime_of(path);
            if (current != last_write)
            {
                last_write = current;
                LoadedImage fresh;
                if (load_ppm_rgb(path, fresh))
                {
                    if (fresh.w != img.w || fresh.h != img.h)
                    {
                        // New dimensions (different scene): rebuild texture.
                        SDL_DestroyTexture(texture);
                        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24,
                                                    SDL_TEXTUREACCESS_STREAMING, fresh.w, fresh.h);
                        if (!texture)
                            break;
                        img.w = fresh.w;
                        img.h = fresh.h;
                    }
                    img.rgb = std::move(fresh.rgb);
                    SDL_UpdateTexture(texture, nullptr, img.rgb.data(), img.w * 3);
                    std::printf("viewer: reloaded %s\n", path.c_str());
                }
            }
        }
        SDL_RenderClear(renderer);
        SDL_RenderTexture(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        if (max_frames > 0 && ++frames >= max_frames)
            running = false;
        SDL_Delay(16);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::printf("viewer: %d frames, bye\n", frames);
    return 0;
}
