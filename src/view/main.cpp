// rt_view: interactive real-time previewer + fly camera + progressive path tracer
// as well as pixel-exact PPM preview + inspector + diff mode.
//
// Usage:
//   Interactive Mode:
//     rt_view [--scene showcase|default|...] [--gpu|--cpu] [--width W] [--height H] [--scale N] [--exposure E]
//     Controls:
//       W/A/S/D       : Fly forward / strafe left / backward / strafe right
//       Space / C     : Fly up / down (also E / Q)
//       Shift / Ctrl  : Sprint (3x speed) / Slow precision sneak (0.25x speed)
//       Right Drag / F: Look around (pitch & yaw) / Toggle captured mouse look
//       Mouse Wheel   : Adjust movement speed
//       G             : Toggle GPU compute / CPU progressive rendering in real time
//       T             : Toggle ACES tonemapping vs standard sRGB gamma
//       [ / ]         : Decrease / Increase exposure
//       R             : Reset camera to initial scene view
//       P             : Print camera parameters to console & save out/viewport.ppm
//       Esc           : Exit mouse capture / Quit
//
//   PPM Inspector Mode (Legacy):
//     rt_view image.ppm [--diff other.ppm] [--scale N] [--stats]
//     Keys: mouse = x/y/RGB readout, D = diff heat, R = reload, Esc = quit.
//
#include "camera/fly_camera.h"
#include "scene/scene.h"
#include "integrator/integrator.h"
#include "accel/qbvh.h"
#include "accel/qbvh8.h"
#include "output/film.h"
#include "output/ppm.h"
#include "gpu/vk_compute.h"
#include "gpu/flatten.h"
#include "gpu/host_scene.h"
#include "io/compare.h"
#include "io/ppm_image.h"
#include "io/denoise.h"
#include "io/bloom.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// Persistent worker thread pool with dynamic work-stealing for interactive viewport
class cpu_render_pool {
public:
    explicit cpu_render_pool(unsigned n) : num_threads(n) {
        for (unsigned t = 0; t < n; ++t) {
            workers.emplace_back([this, t] {
                int local_gen = 0;
                while (true) {
                    std::function<void(unsigned)> fn;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv_start.wait(lock, [this, local_gen] {
                            return stop.load(std::memory_order_relaxed) || (start_gen.load(std::memory_order_relaxed) > local_gen);
                        });
                        if (stop.load(std::memory_order_relaxed))
                            return;
                        local_gen = start_gen.load(std::memory_order_relaxed);
                        fn = worker_fn;
                    }
                    if (fn)
                        fn(t);
                    if (done_count.fetch_add(1, std::memory_order_acq_rel) + 1 == (int)num_threads) {
                        cv_done.notify_one();
                    }
                }
            });
        }
    }

    void parallel_run(std::function<void(unsigned)> fn) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            worker_fn = std::move(fn);
            done_count.store(0, std::memory_order_relaxed);
            start_gen.fetch_add(1, std::memory_order_release);
        }
        cv_start.notify_all();
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_done.wait(lock, [this] {
                return done_count.load(std::memory_order_acquire) == (int)num_threads;
            });
        }
    }

    ~cpu_render_pool() {
        stop.store(true, std::memory_order_release);
        cv_start.notify_all();
        for (auto &w : workers) {
            if (w.joinable())
                w.join();
        }
    }

private:
    unsigned num_threads;
    std::vector<std::thread> workers;
    std::atomic<bool> stop{false};
    std::mutex mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    std::atomic<int> start_gen{0};
    std::atomic<int> done_count{0};
    std::function<void(unsigned)> worker_fn;
};

// =============================================================================
// Legacy PPM Viewer Functions
// =============================================================================

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
    mean /= (double)px.size();
    std::cout << label << " " << w << "x" << h << " mean=" << mean << "\n";
}

int run_image_viewer(int argc, char **argv) {
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

// =============================================================================
// Interactive Real-Time Previewer (Fly Camera + Progressive Path Tracer)
// =============================================================================

int run_interactive_renderer(int argc, char **argv) {
    std::string scene_name = "showcase";
    int W = 800;
    int H = 450;
    int scale = 1;
    bool force_cpu = false;
    bool force_gpu = false;
    double exposure = 1.0;
    int max_depth = 8; // fast bounce depth for responsive interactive navigation
    double aperture = 0.0;
    std::string hdri_env_path;
    bool show_hud = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scene" && i + 1 < argc)
            scene_name = argv[++i];
        else if (a == "--width" && i + 1 < argc)
            W = std::max(64, std::atoi(argv[++i]));
        else if (a == "--height" && i + 1 < argc)
            H = std::max(36, std::atoi(argv[++i]));
        else if (a == "--scale" && i + 1 < argc)
            scale = std::max(1, std::atoi(argv[++i]));
        else if (a == "--exposure" && i + 1 < argc)
            exposure = std::atof(argv[++i]);
        else if (a == "--maxdepth" && i + 1 < argc)
            max_depth = std::max(1, std::atoi(argv[++i]));
        else if (a == "--aperture" && i + 1 < argc)
            aperture = std::atof(argv[++i]);
        else if (a == "--hdri-env" && i + 1 < argc)
            hdri_env_path = argv[++i];
        else if (a == "--cpu")
            force_cpu = true;
        else if (a == "--gpu")
            force_gpu = true;
        else if (a == "--no-hud")
            show_hud = false;
    }

    std::cout << "rt_view: Initializing interactive scene '" << scene_name
              << "' (" << W << "x" << H << ", scale=" << scale << ")...\n";

    double aspect = (double)W / (double)H;
    scene_data sdata = build_scene(scene_name, aspect, aperture);
    if (!hdri_env_path.empty()) {
        sdata.hdri = std::make_shared<hdri_env>();
        if (!sdata.hdri->load(hdri_env_path)) {
            std::cerr << "hdri-env: could not load '" << hdri_env_path << "'\n";
            sdata.hdri.reset();
        } else {
            sdata.env_light = true;
            sdata.black_bg = false;
        }
    }

    // Set up fly camera
    double initial_vfov = (aspect < 1.0) ? 52.0 : 37.0;
    if (scene_name == "default") initial_vfov = 20.0;
    fly_camera fly_cam;
    fly_cam.init_from_camera(sdata.cam, initial_vfov);
    double focus_dist = 4.0;
    camera active_cam = fly_cam.build_camera(aspect, aperture, focus_dist, 0, 1.0, 0.0);

    // Build CPU QBVH (fp32 8-wide) + power-CDF for NEE.
    std::cout << "rt_view: Building CPU QBVH acceleration structure (" << sdata.objs.size() << " primitives)...\n";
    qbvh8_node world(sdata.objs, 0, sdata.objs.size());
    std::vector<double> view_cdf;
    double view_total = 0.0;
    if (!sdata.lights.empty())
        build_light_cdf(sdata.lights, view_cdf, view_total);
    integrator tracer;
    render_params params{world, sdata.lights, max_depth, sdata.media,
                         sdata.env_light, sdata.black_bg};
    params.light_cdf = view_cdf.empty() ? nullptr : &view_cdf;
    params.light_power_total = view_total;

    // Try GPU initialization
    bool use_gpu = false;
    GpuContext gpu{};
    flat_scene flat;
    std::string shader_path = SHADER_DIR "/path.spv";
    if (!std::filesystem::exists(shader_path)) {
        if (std::filesystem::exists("build/shaders/path.spv"))
            shader_path = "build/shaders/path.spv";
        else if (std::filesystem::exists("shaders/path.spv"))
            shader_path = "shaders/path.spv";
    }

    PushConstants push{};
    std::vector<float> gpu_rgba;

    if (!force_cpu && std::filesystem::exists(shader_path)) {
        std::cout << "rt_view: Flattening scene for GPU compute...\n";
        if (flatten_scene(sdata, flat)) {
            gpu_init(gpu, W, H);
            gpu_scene &gscene = flat.gs;
            gscene.cam = gpu_cam_from_cpu(active_cam.eye(), active_cam.corner(),
                                          active_cam.span_u(), active_cam.span_v(),
                                          active_cam.lens_r(), 0,
                                          1.0, 0.0);

            std::vector<int> img_table;
            std::vector<float> img_blob;
            for (const auto &im : flat.images) {
                img_table.push_back((int)img_blob.size() / 4);
                img_table.push_back(im.w);
                img_table.push_back(im.h);
                img_table.push_back(im.levels);
                int spanbits = 0;
                static_assert(sizeof(spanbits) == sizeof(im.span));
                std::memcpy(&spanbits, &im.span, sizeof spanbits);
                img_table.push_back(spanbits);
                img_table.push_back(0);
                img_table.push_back(0);
                img_table.push_back(0);
                img_blob.insert(img_blob.end(), im.rgba.begin(), im.rgba.end());
            }
            static const float empty_blob[4] = {};
            const void *img_ptr = img_blob.empty() ? empty_blob : img_blob.data();
            size_t img_bytes = img_blob.empty() ? sizeof empty_blob : img_blob.size() * sizeof(float);
            static const int empty_tab[4] = {};
            const void *tab_ptr = img_table.empty() ? empty_tab : img_table.data();
            size_t tab_bytes = img_table.empty() ? sizeof empty_tab : img_table.size() * sizeof(int);

            std::vector<int> light_tab;
            for (const auto &le : flat.light_table) {
                light_tab.push_back(le.first);
                light_tab.push_back(le.second);
            }
            static const int empty_light[2] = {};
            const void *light_ptr = light_tab.empty() ? empty_light : light_tab.data();
            size_t light_bytes = light_tab.empty() ? sizeof empty_light : light_tab.size() * sizeof(int);

            static const float empty_hdri[4] = {};
            std::vector<float> hdri_tex_buf, hdri_cdf_buf;
            if (sdata.hdri && !sdata.hdri->empty()) {
                hdri_tex_buf = sdata.hdri->texel_upload();
                hdri_cdf_buf = sdata.hdri->cdf_upload();
            }
            const void *hdri_tex_ptr = hdri_tex_buf.empty() ? (const void*)empty_hdri : (const void*)hdri_tex_buf.data();
            size_t hdri_tex_bytes = hdri_tex_buf.empty() ? sizeof empty_hdri : hdri_tex_buf.size() * sizeof(float);
            const void *hdri_cdf_ptr = hdri_cdf_buf.empty() ? (const void*)empty_hdri : (const void*)hdri_cdf_buf.data();
            size_t hdri_cdf_bytes = hdri_cdf_buf.empty() ? sizeof empty_hdri : hdri_cdf_buf.size() * sizeof(float);
            // Light power CDF over the GPU light table (see flatten.h).
            std::vector<float> light_cdf_buf;
            {
                float gtotal = 0;
                build_gpu_light_cdf(flat, light_cdf_buf, gtotal);
                light_cdf_buf.push_back(gtotal);
            }
            static const float empty_cdf[1] = {};
            const void *light_cdf_ptr =
                light_cdf_buf.size() > 1 ? (const void *)light_cdf_buf.data() : (const void *)empty_cdf;
            size_t light_cdf_bytes =
                light_cdf_buf.size() > 1 ? light_cdf_buf.size() * sizeof(float) : sizeof empty_cdf;

            const void *data[12] = {&gscene.cam, gscene.spheres.data(), gscene.quads.data(),
                                    gscene.tris.data(), flat.nodes.data(), flat.refs.data(),
                                    img_ptr, tab_ptr, light_ptr, hdri_tex_ptr, hdri_cdf_ptr,
                                    light_cdf_ptr};
            const size_t bytes[12] = {sizeof gscene.cam, gscene.spheres.size() * sizeof(GPUSphere),
                                      gscene.quads.size() * sizeof(GPUQuad),
                                      gscene.tris.size() * sizeof(GPUTri),
                                      flat.nodes.size() * sizeof(GPUQNode),
                                      flat.refs.size() * sizeof(GPURef), img_bytes, tab_bytes,
                                      light_bytes, hdri_tex_bytes, hdri_cdf_bytes,
                                      light_cdf_bytes};
            gpu_set_scene(gpu, data, bytes);

            int nfog = 0;
            for (const auto &s : gscene.spheres)
                if (s.prm[0] == static_cast<float>(MatType::FOG) || s.prm[0] == static_cast<float>(MatType::HET))
                    nfog++;

            push.W = W;
            push.H = H;
            push.ns = (int)gscene.spheres.size();
            push.nq = (int)gscene.quads.size();
            push.nt = (int)gscene.tris.size();
            push.spp = 1;
            push.seed = 1;
            push.nlights = (int)flat.nlights;
            push.sh0 = 0.0f;
            push.sh1 = 0.0f;
            push.nfog = nfog;
            push.nenv = sdata.nenv_mode();
            push.nblack = sdata.black_bg ? 1 : 0;
            push.maxdepth = max_depth;

            use_gpu = true;
            std::cout << "rt_view: GPU compute active (" << shader_path << ").\n";
        }
    }

    if (!use_gpu && force_gpu) {
        std::cerr << "rt_view: Warning: GPU acceleration requested but unavailable, falling back to CPU.\n";
    }

    // Initialize SDL3 Window
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init: " << SDL_GetError() << "\n";
        return 1;
    }
    std::string win_title = "rt_view [" + std::string(use_gpu ? "GPU" : "CPU") + "] " + scene_name;
    SDL_Window *win = SDL_CreateWindow(win_title.c_str(), W * scale, H * scale, SDL_WINDOW_RESIZABLE);
    if (!win) {
        std::cerr << "window: " << SDL_GetError() << "\n";
        return 1;
    }
    SDL_Renderer *ren = SDL_CreateRenderer(win, nullptr);
    SDL_Texture *tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB24,
                                         SDL_TEXTUREACCESS_STREAMING, W, H);
    SDL_SetTextureScaleMode(tex, SDL_SCALEMODE_NEAREST);

    bool accum_paused = false;

    // Buffers for progressive accumulation and AOV inspection
    std::vector<vec3> accum_fb((size_t)W * H, vec3(0, 0, 0));
    std::vector<vec3> beauty_hdr((size_t)W * H, vec3(0, 0, 0));
    std::vector<uint8_t> display_rgb((size_t)W * H * 3, 0);
    // Present path (GPU backend, plain beauty): on-device accumulation +
    // film straight to display bytes; no float downloads, no host tonemap.
    PresentAccum present{};
    std::vector<unsigned char> present_bytes;
    std::string shdir =
        shader_path.substr(0, shader_path.find_last_of("/\\") + 1);
    bool present_clear = true;
    bool present_active = false;
    int accum_spp = 0;
    bool use_aces = true;
    bool relative_mouse = false;
    bool right_mouse_down = false;
    bool live_denoise = false;
    bool bloom_enabled = false;

    unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    cpu_render_pool pool(num_threads);

    auto last_frame_time = std::chrono::steady_clock::now();
    auto last_title_time = last_frame_time;
    int frames_since_title = 0;
    double current_fps = 0.0;
    double current_ms = 0.0;

    bool running = true;
    char title_buf[256];

    std::cout << "\n=== Interactive Controls ===\n"
              << "  WASD         : Fly forward / backward / strafe left / right\n"
              << "  Space / C    : Fly up / down (also E / Q)\n"
              << "  Shift / Ctrl : Sprint (3x) / Sneak (0.25x precision)\n"
              << "  Right Drag/F : Mouse Look / Toggle Cursor Lock\n"
              << "  Mouse Wheel  : Adjust flight speed (current: " << fly_cam.speed << ")\n"
              << "  Tab / F11    : Toggle On-Screen Heads-Up Display (HUD) [ON/OFF]\n"
              << "  N            : Toggle Live Bilateral Denoiser [ON/OFF]\n"
              << "  M            : Toggle Multi-Scale Bloom [ON/OFF]\n"
              << "  X            : Toggle Accumulation Pause / Resume\n"
              << "  G            : Toggle GPU compute vs CPU multi-threading\n"
              << "  T            : Toggle ACES film tonemapping vs Linear/sRGB\n"
              << "  [ / ]        : Adjust Exposure (current: " << exposure << ")\n"
              << "  R            : Reset Camera to origin\n"
              << "  P / F12      : Save Viewport PPM & Print Camera Code\n"
              << "  H            : Print Interactive Hotkey Guide\n"
              << "  Esc          : Exit Viewport\n\n";

    while (running) {
        auto frame_start = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(frame_start - last_frame_time).count();
        last_frame_time = frame_start;
        // Clamp dt to avoid huge jumps on window drag
        if (dt > 0.1) dt = 0.1;

        bool cam_moved = false;

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (e.type == SDL_EVENT_KEY_DOWN) {
                if (e.key.key == SDLK_ESCAPE) {
                    if (relative_mouse) {
                        relative_mouse = false;
                        SDL_SetWindowRelativeMouseMode(win, false);
                    } else {
                        running = false;
                    }
                } else if (e.key.key == SDLK_F) {
                    relative_mouse = !relative_mouse;
                    SDL_SetWindowRelativeMouseMode(win, relative_mouse);
                } else if (e.key.key == SDLK_G) {
                    if (gpu.has_scene) {
                        use_gpu = !use_gpu;
                        std::cout << "rt_view: Switched backend to [" << (use_gpu ? "GPU" : "CPU") << "]\n";
                        cam_moved = true;
                    } else {
                        std::cout << "rt_view: GPU compute not available for this session.\n";
                    }
                } else if (e.key.key == SDLK_T) {
                    use_aces = !use_aces;
                    std::cout << "rt_view: Tonemapping set to [" << (use_aces ? "ACES" : "Linear/sRGB") << "]\n";
                } else if (e.key.key == SDLK_LEFTBRACKET) {
                    exposure = std::max(0.05, exposure * 0.8);
                    std::cout << "rt_view: Exposure = " << exposure << "\n";
                } else if (e.key.key == SDLK_RIGHTBRACKET) {
                    exposure = std::min(50.0, exposure * 1.25);
                    std::cout << "rt_view: Exposure = " << exposure << "\n";
                } else if (e.key.key == SDLK_N) {
                    live_denoise = !live_denoise;
                    std::cout << "rt_view: Live Bilateral AOV Denoiser [" << (live_denoise ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_M) {
                    bloom_enabled = !bloom_enabled;
                    std::cout << "rt_view: Multi-Scale Bloom & Optical Glare [" << (bloom_enabled ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_TAB || e.key.key == SDLK_F11) {
                    show_hud = !show_hud;
                    std::cout << "rt_view: Heads-Up Display (HUD) [" << (show_hud ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_R) {
                    fly_cam.init_from_camera(sdata.cam, initial_vfov);
                    focus_dist = 4.0;
                    cam_moved = true;
                    std::cout << "rt_view: Camera reset to origin.\n";
                } else if (e.key.key == SDLK_X) {
                    accum_paused = !accum_paused;
                    std::cout << "rt_view: Accumulation " << (accum_paused ? "[PAUSED]" : "[RESUMED]") << "\n";
                } else if (e.key.key == SDLK_H) {
                    std::cout << "\n=== rt_view Interactive Controls ===\n"
                              << " WASD         : Fly forward / backward / strafe left / right\n"
                              << " Space / C    : Fly vertically up / down (also E / Q)\n"
                              << " Shift/Ctrl   : Sprint 3x / Sneak 0.25x precision\n"
                              << " Right Drag   : Look around (pitch & yaw)\n"
                              << " F            : Toggle captured mouse look mode\n"
                              << " Mouse Wheel  : Adjust movement speed\n"
                              << " Tab / F11    : Toggle On-Screen Heads-Up Display (HUD)\n"
                              << " N            : Toggle Live Bilateral Denoising\n"
                              << " M            : Toggle Multi-Scale Bloom\n"
                              << " X            : Toggle Accumulation Pause / Resume\n"
                              << " G            : Toggle GPU compute vs CPU multi-threading\n"
                              << " T            : Toggle ACES film tonemapping vs Linear/sRGB\n"
                              << " [ / ]        : Exposure decrease / increase\n"
                              << " R            : Reset camera to origin\n"
                              << " P            : Save viewport snapshot to out/viewport.ppm\n"
                              << " Esc          : Exit viewport\n\n";
                } else if (e.key.key == SDLK_P || e.key.key == SDLK_F12) {
                    std::cout << "\n// Camera Snapshot (SPP: " << accum_spp << "):\n"
                              << "vec3 lookfrom(" << fly_cam.eye.x() << ", "
                              << fly_cam.eye.y() << ", " << fly_cam.eye.z() << ");\n"
                              << "vec3 lookat(" << (fly_cam.eye + fly_cam.forward_dir()).x() << ", "
                              << (fly_cam.eye + fly_cam.forward_dir()).y() << ", "
                              << (fly_cam.eye + fly_cam.forward_dir()).z() << ");\n"
                              << "double vfov = " << fly_cam.vfov << ";\n"
                              << "double focus_dist = " << focus_dist << ";\n"
                              << "double aperture = " << aperture << ";\n"
                              << "// yaw = " << fly_cam.yaw << ", pitch = " << fly_cam.pitch << "\n";

                    std::filesystem::create_directories("out");
                    if (present_active && accum_spp > 0 && present_bytes.size() == (size_t)W * H * 4) {
                        // Present bytes are already filmed (top-first RGBA):
                        // emit P6 directly.
                        std::string buf;
                        buf.reserve((size_t)W * H * 3 + 32);
                        buf.append("P6\n" + std::to_string(W) + " " + std::to_string(H) + "\n255\n");
                        for (size_t i = 0; i < (size_t)W * H; ++i) {
                            buf.push_back((char)present_bytes[i * 4 + 0]);
                            buf.push_back((char)present_bytes[i * 4 + 1]);
                            buf.push_back((char)present_bytes[i * 4 + 2]);
                        }
                        std::ofstream out("out/viewport.ppm", std::ios::binary);
                        if (out) {
                            out.write(buf.data(), (std::streamsize)buf.size());
                            std::cout << "rt_view: Saved current viewport to out/viewport.ppm\n\n";
                        }
                    } else {
                        // Flip bottom-first for standard PPM
                        std::vector<vec3> ppm_fb((size_t)W * H);
                        double inv = (accum_spp > 0) ? (1.0 / accum_spp) : 1.0;
                        for (int y = 0; y < H; ++y)
                            for (int x = 0; x < W; ++x)
                                ppm_fb[((size_t)H - 1 - y) * W + x] = accum_fb[(size_t)y * W + x] * inv;
                        if (write_ppm("out/viewport.ppm", ppm_fb, W, H, exposure))
                            std::cout << "rt_view: Saved current viewport to out/viewport.ppm\n\n";
                    }
                }
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (e.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down = true;
                    SDL_SetWindowRelativeMouseMode(win, true);
                }
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                if (e.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down = false;
                    if (!relative_mouse)
                        SDL_SetWindowRelativeMouseMode(win, false);
                }
            } else if (e.type == SDL_EVENT_MOUSE_MOTION) {
                if (relative_mouse || right_mouse_down) {
                    if (fly_cam.process_mouse((double)e.motion.xrel, (double)e.motion.yrel))
                        cam_moved = true;
                }
            } else if (e.type == SDL_EVENT_MOUSE_WHEEL) {
                if (e.wheel.y > 0)
                    fly_cam.speed = std::min(50.0, fly_cam.speed * 1.25);
                else if (e.wheel.y < 0)
                    fly_cam.speed = std::max(0.2, fly_cam.speed * 0.8);
            }
        }

        // Process keyboard movement
        const bool *ks = SDL_GetKeyboardState(nullptr);
        if (ks) {
            bool fwd = ks[SDL_SCANCODE_W];
            bool bwd = ks[SDL_SCANCODE_S];
            bool lft = ks[SDL_SCANCODE_A];
            bool rgt = ks[SDL_SCANCODE_D];
            bool up  = ks[SDL_SCANCODE_SPACE] || ks[SDL_SCANCODE_E];
            bool dn  = ks[SDL_SCANCODE_C]     || ks[SDL_SCANCODE_Q];
            bool sprint = ks[SDL_SCANCODE_LSHIFT] || ks[SDL_SCANCODE_RSHIFT];
            bool slow   = ks[SDL_SCANCODE_LCTRL]  || ks[SDL_SCANCODE_RCTRL];

            if (fly_cam.process_keyboard(dt, fwd, bwd, lft, rgt, up, dn, sprint, slow))
                cam_moved = true;
        }

        // Present mode needs host pixels only for denoise/bloom; plain
        // beauty films on device. Switching modes resets accumulation.
        bool want_present = use_gpu && !live_denoise && !bloom_enabled;
        if (want_present != present_active && accum_spp > 0) {
            accum_spp = 0;
            std::fill(accum_fb.begin(), accum_fb.end(), vec3(0, 0, 0));
            present_clear = true;
        }
        present_active = want_present;
        // Reset accumulation on camera movement
        if (cam_moved) {
            accum_spp = 0;
            std::fill(accum_fb.begin(), accum_fb.end(), vec3(0, 0, 0));
            present_clear = true;
            active_cam = fly_cam.build_camera(aspect, aperture, focus_dist, 0, 1.0, 0.0);
            if (gpu.has_scene) {
                GPUCam gcam = gpu_cam_from_cpu(active_cam.eye(), active_cam.corner(),
                                               active_cam.span_u(), active_cam.span_v(),
                                               active_cam.lens_r(), 0,
                                               1.0, 0.0);
                gpu_update_camera(gpu, &gcam, sizeof(gcam));
            }
        }

        // Render pass: dispatch progressive path tracing when not paused
        if (!accum_paused) {
            if (use_gpu && present_active) {
                // Present fast path: chunk stays on device, folded into the
                // persistent sum; zero float downloads.
                push.spp = 1;
                push.seed = (uint32_t)accum_spp + 1;
                gpu_run(gpu, shader_path, push, gpu_rgba, true);
                present_accum_add(gpu, present, shdir + "accum.spv",
                                  present_clear);
                present_clear = false;
                accum_spp += 1;
            } else if (use_gpu) {
                push.spp = 1;
                push.seed = (uint32_t)accum_spp + 1;
                gpu_run(gpu, shader_path, push, gpu_rgba);
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    accum_fb[i] += vec3(gpu_rgba[i * 4 + 0], gpu_rgba[i * 4 + 1], gpu_rgba[i * 4 + 2]);
                }
                accum_spp += 1;
            } else {
                // Multi-threaded CPU progressive slice with persistent pool & dynamic scanline work-stealing
                std::atomic<int> next_row{0};
                pool.parallel_run([&](unsigned t) {
                    rng_seed(42u + (unsigned)accum_spp * 10007u + t * 997u);
                    for (;;) {
                        int sy = next_row.fetch_add(1, std::memory_order_relaxed);
                        if (sy >= H)
                            break;
                        int j = H - 1 - sy; // Screen row 0 is top; ray v=0 is bottom
                        for (int i = 0; i < W; ++i) {
                            double u = (i + random_double()) / (double)W;
                            double v = (j + random_double()) / (double)H;
                            ray r = active_cam.get_ray(u, v);
                            vec3 col = tracer.Li(r, params);
                            accum_fb[(size_t)sy * W + i] += col;
                        }
                    }
                });
                accum_spp += 1;
            }
        }

        // Convert accumulated beauty to display RGB24
        if (present_active && accum_spp > 0 && gpu.has_scene) {
            // Device film: sum/count -> exposure -> ACES/sRGB -> bytes.
            present_tonemap(gpu, present, shdir + "tonemap.spv",
                            (float)exposure, use_aces ? 1 : 0,
                            (float)accum_spp, present_bytes);
            for (size_t i = 0; i < (size_t)W * H; ++i) {
                display_rgb[i * 3 + 0] = present_bytes[i * 4 + 0];
                display_rgb[i * 3 + 1] = present_bytes[i * 4 + 1];
                display_rgb[i * 3 + 2] = present_bytes[i * 4 + 2];
            }
        } else {
            double inv_spp = (accum_spp > 0) ? (1.0 / (double)accum_spp) : 1.0;
            std::atomic<int> next_row_hdr{0};
            pool.parallel_run([&](unsigned) {
                for (;;) {
                    int y = next_row_hdr.fetch_add(1, std::memory_order_relaxed);
                    if (y >= H)
                        break;
                    size_t row_offset = (size_t)y * W;
                    for (int x = 0; x < W; ++x)
                        beauty_hdr[row_offset + x] = accum_fb[row_offset + x] * inv_spp;
                }
            });
            if (live_denoise && accum_spp >= 2)
                beauty_hdr = bilateral_denoise(beauty_hdr, W, H, 1.5, 0.15);
            if (bloom_enabled)
                beauty_hdr = bloom::apply_bloom(beauty_hdr, W, H, 1.0, 0.08);
            std::atomic<int> next_row_disp{0};
            pool.parallel_run([&](unsigned) {
                for (;;) {
                    int y = next_row_disp.fetch_add(1, std::memory_order_relaxed);
                    if (y >= H)
                        break;
                    for (int x = 0; x < W; ++x) {
                        size_t i = (size_t)y * W + x;
                        vec3 hdr = beauty_hdr[i];
                        vec3 ldr = use_aces ? tonemap(hdr, exposure)
                                            : vec3(srgb_encode(hdr.x() * exposure),
                                                   srgb_encode(hdr.y() * exposure),
                                                   srgb_encode(hdr.z() * exposure));
                        uint8_t r = (uint8_t)(std::clamp(ldr.x(), 0.0, 1.0) * 255.999);
                        uint8_t g = (uint8_t)(std::clamp(ldr.y(), 0.0, 1.0) * 255.999);
                        uint8_t b = (uint8_t)(std::clamp(ldr.z(), 0.0, 1.0) * 255.999);
                        display_rgb[i * 3 + 0] = r;
                        display_rgb[i * 3 + 1] = g;
                        display_rgb[i * 3 + 2] = b;
                    }
                }
            });
        }

        SDL_UpdateTexture(tex, nullptr, display_rgb.data(), W * 3);
        SDL_RenderClear(ren);
        SDL_RenderTexture(ren, tex, nullptr, nullptr);

        // On-Screen Heads-Up Display (HUD) Glassmorphism Overlay
        if (show_hud) {
            int win_w = 0, win_h = 0;
            SDL_GetRenderOutputSize(ren, &win_w, &win_h);
            if (win_w <= 0) win_w = W * scale;
            if (win_h <= 0) win_h = H * scale;

            float hud_x = 14.0f;
            float hud_y = 14.0f;
            float hud_w = std::min((float)win_w - 28.0f, 490.0f);
            float hud_h = 196.0f;

            // Translucent glassmorphism background
            SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
            SDL_FRect bg_rect{ hud_x, hud_y, hud_w, hud_h };
            SDL_SetRenderDrawColor(ren, 10, 14, 22, 215); // Deep slate
            SDL_RenderFillRect(ren, &bg_rect);

            // Subtle border
            SDL_SetRenderDrawColor(ren, 45, 80, 115, 220);
            SDL_RenderRect(ren, &bg_rect);

            float text_x = hud_x + 12.0f;
            float text_y = hud_y + 10.0f;
            float line_step = 13.5f;
            char line_buf[256];

            // 1. Header Title & Backend
            SDL_SetRenderDrawColor(ren, 50, 220, 255, 255); // Cyan
            std::snprintf(line_buf, sizeof(line_buf), "RT_VIEW :: %s", scene_name.c_str());
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);

            const char *backend_str = use_gpu ? "[GPU: Vulkan Path Tracer]" : "[CPU: Multithreaded QBVH]";
            SDL_SetRenderDrawColor(ren, use_gpu ? 90 : 255, use_gpu ? 240 : 190, use_gpu ? 140 : 70, 255);
            SDL_RenderDebugText(ren, hud_x + hud_w - 215.0f, text_y, backend_str);
            text_y += line_step + 2.0f;

            // Separator line
            SDL_SetRenderDrawColor(ren, 40, 65, 95, 180);
            SDL_RenderLine(ren, hud_x + 8.0f, text_y, hud_x + hud_w - 8.0f, text_y);
            text_y += 5.0f;

            // 2. Performance & Accumulation
            SDL_SetRenderDrawColor(ren, 120, 255, 140, 255); // Lime
            std::snprintf(line_buf, sizeof(line_buf), "PERF: %.1f FPS (%.1f ms/frame)", current_fps, current_ms);
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);

            // SPP & Scale
            SDL_SetRenderDrawColor(ren, accum_paused ? 255 : 240, accum_paused ? 180 : 245, accum_paused ? 50 : 255, 255);
            std::snprintf(line_buf, sizeof(line_buf), "SPP : %d %s",
                          accum_spp, accum_paused ? "[PAUSED]" : "");
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);
            text_y += line_step + 2.0f;

            // 3. Camera & Optics
            SDL_SetRenderDrawColor(ren, 170, 190, 215, 255);
            std::snprintf(line_buf, sizeof(line_buf), "EYE : (%.2f, %.2f, %.2f) | FOV: %.1f deg",
                          fly_cam.eye.x(), fly_cam.eye.y(), fly_cam.eye.z(), fly_cam.vfov);
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);
            text_y += line_step;

            std::snprintf(line_buf, sizeof(line_buf), "ROT : Yaw %.1f deg | Pitch %.1f deg",
                          fly_cam.yaw, fly_cam.pitch);
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);
            text_y += line_step;

            std::snprintf(line_buf, sizeof(line_buf), "LENS: Focus %.2fm | Aperture f/%.2f",
                          focus_dist, aperture);
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);
            text_y += line_step + 2.0f;

            // 4. Post-FX
            SDL_SetRenderDrawColor(ren, 175, 195, 220, 255);
            std::snprintf(line_buf, sizeof(line_buf), "FX  : Bloom:%s Denoise:%s",
                          bloom_enabled ? "ON" : "OFF",
                          live_denoise ? "ON" : "OFF");
            SDL_RenderDebugText(ren, text_x, text_y, line_buf);
            text_y += line_step;

            // Bottom quick hint ribbon
            if (win_h > 260) {
                float bar_h = 22.0f;
                float bar_y = (float)win_h - bar_h - 8.0f;
                SDL_FRect bar_bg{ 14.0f, bar_y, (float)win_w - 28.0f, bar_h };
                SDL_SetRenderDrawColor(ren, 10, 14, 22, 195);
                SDL_RenderFillRect(ren, &bar_bg);
                SDL_SetRenderDrawColor(ren, 40, 65, 95, 160);
                SDL_RenderRect(ren, &bar_bg);

                SDL_SetRenderDrawColor(ren, 185, 205, 225, 255);
                SDL_RenderDebugText(ren, 22.0f, bar_y + 7.0f,
                                    "[Tab] HUD  [WASD] Fly  [Right-Drag] Look  [P] Snapshot  [H] Help");
            }
        }

        SDL_RenderPresent(ren);

        // Calculate and update live performance metrics
        frames_since_title++;
        auto frame_end = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(frame_end - last_title_time).count();
        if (elapsed_sec >= 0.15) {
            current_fps = (double)frames_since_title / elapsed_sec;
            current_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            frames_since_title = 0;
            last_title_time = frame_end;

            std::snprintf(title_buf, sizeof(title_buf),
                          "rt_view [%s] %s%s | SPP: %d | %.1f FPS (%.1f ms) | Den: %s | Bloom: %s",
                          use_gpu ? "GPU" : "CPU", scene_name.c_str(),
                          accum_paused ? " [PAUSED]" : "", accum_spp,
                          current_fps, current_ms,
                          (live_denoise ? "ON" : "OFF"),
                          (bloom_enabled ? "ON" : "OFF"));
            SDL_SetWindowTitle(win, title_buf);
        }
    }

    if (gpu.has_scene) {
        present_accum_destroy(gpu, present);
        gpu_shutdown(gpu);
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    bool interactive = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--scene" || a == "-i" || a == "--interactive" || a == "--gpu" || a == "--cpu")
            interactive = true;
        if (a.ends_with(".ppm") || a == "--diff" || a == "--stats")
            interactive = false;
    }
    // Launch interactive showcase when called without arguments
    if (argc == 1)
        interactive = true;

    if (interactive)
        return run_interactive_renderer(argc, argv);
    else
        return run_image_viewer(argc, argv);
}
