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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

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
    int blades = 0;
    double anamorphic = 1.0;
    double distortion = 0.0;
    camera active_cam = fly_cam.build_camera(aspect, aperture, focus_dist, blades, anamorphic, distortion);

    // Build CPU QBVH
    std::cout << "rt_view: Building CPU QBVH acceleration structure (" << sdata.objs.size() << " primitives)...\n";
    qbvh_node world(sdata.objs, 0, sdata.objs.size(), true);
    integrator tracer;
    render_params params{world, sdata.lights, max_depth, sdata.media,
                         sdata.env_light, sdata.black_bg, false, false};

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
                                          active_cam.lens_r(), blades,
                                          anamorphic, distortion);

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

            const void *data[11] = {&gscene.cam, gscene.spheres.data(), gscene.quads.data(),
                                    gscene.tris.data(), flat.nodes.data(), flat.refs.data(),
                                    img_ptr, tab_ptr, light_ptr, hdri_tex_ptr, hdri_cdf_ptr};
            const size_t bytes[11] = {sizeof gscene.cam, gscene.spheres.size() * sizeof(GPUSphere),
                                      gscene.quads.size() * sizeof(GPUQuad),
                                      gscene.tris.size() * sizeof(GPUTri),
                                      flat.nodes.size() * sizeof(GPUQNode),
                                      flat.refs.size() * sizeof(GPURef), img_bytes, tab_bytes,
                                      light_bytes, hdri_tex_bytes, hdri_cdf_bytes};
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

    enum class ViewMode { BEAUTY = 0, ALBEDO = 1, NORMAL = 2 };
    ViewMode view_mode = ViewMode::BEAUTY;
    bool accum_paused = false;

    // Buffers for progressive accumulation and AOV inspection
    std::vector<vec3> accum_fb((size_t)W * H, vec3(0, 0, 0));
    std::vector<uint8_t> display_rgb((size_t)W * H * 3, 0);
    std::vector<uint8_t> prev_display_rgb((size_t)W * H * 3, 0);
    std::vector<float> aov_alb((size_t)W * H * 4, 0.0f);
    std::vector<float> aov_nrm((size_t)W * H * 4, 0.0f);
    int accum_spp = 0;
    bool use_aces = true;
    bool relative_mouse = false;
    bool right_mouse_down = false;
    bool temporal_smooth = true;
    bool live_denoise = false;
    bool optical_vignette = false;
    bool bloom_enabled = false;
    bool chromatic_aberration = false;
    bool focus_peaking = false;
    bool turntable_mode = false;
    double turntable_rpm = 2.5;
    double color_temp = 0.0;
    std::shared_ptr<material> selected_mat = nullptr;

    unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());

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
              << "  Mid Click/F4 : Click-to-Focus / Center Autofocus (sets focal plane)\n"
              << "  Alt + Left   : Material & Object Inspector (queries primitive under cursor)\n"
              << "  U / I / O    : Adjust Aperture (U: -0.02, I: +0.02, O: Toggle pinhole/bokeh)\n"
              << "  K / L        : Adjust Focus Distance (K: -0.2m, L: +0.2m)\n"
              << "  B            : Toggle Bokeh Iris Shape (Circular <-> 6-Blade Hexagon)\n"
              << "  J            : Toggle Anamorphic Lens Squeeze (1.0x Spherical <-> 2.0x Oval)\n"
              << "  Y            : Toggle Radial Lens Distortion (0.0 Rectilinear <-> 0.20 Barrel)\n"
              << "  9 / F5       : Toggle Lens Chromatic Aberration [ON/OFF]\n"
              << "  F6           : Toggle Focus Peaking / Cinema Z-Peaking Overlay [ON/OFF]\n"
              << "  F8           : Toggle Cinematic 360 Turntable Orbit [ON/OFF]\n"
              << "  Arrow Keys   : Dynamic Sun Orbit (Left/Right: Azimuth, Up/Down: Elevation)\n"
              << "  - / =        : Nudge Roughness (-0.05 / +0.05) or Light Emission (0.8x / 1.25x)\n"
              << "  , / .        : Nudge Glass IOR (-0.05 / +0.05) or Light Temperature (-500K / +500K)\n"
              << "  Z            : Toggle Temporal Motion Smoothing [ON/OFF]\n"
              << "  N            : Toggle Live Bilateral AOV Denoiser [ON/OFF]\n"
              << "  V            : Toggle Optical Vignetting [ON/OFF]\n"
              << "  M            : Toggle Multi-Scale Bloom & Optical Glare [ON/OFF]\n"
              << "  ; / ' / /    : Color Temperature ( ; Cooler, ' Warmer, / Reset )\n"
              << "  1 - 4        : Camera bookmarks (Bunny, Crystals, Disney, Wide)\n"
              << "  F1 - F3      : Display Mode (F1: Beauty, F2: Albedo AOV, F3: Normal AOV)\n"
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

        if (turntable_mode) {
            double d_angle_deg = (turntable_rpm * 360.0 / 60.0) * dt;
            vec3 target = fly_cam.eye + fly_cam.forward_dir() * focus_dist;
            double rad = d_angle_deg * (3.1415926535897932385 / 180.0);
            double dx = fly_cam.eye.x() - target.x();
            double dz = fly_cam.eye.z() - target.z();
            double cos_r = std::cos(rad), sin_r = std::sin(rad);
            double new_dx = dx * cos_r - dz * sin_r;
            double new_dz = dx * sin_r + dz * cos_r;
            fly_cam.eye = vec3(target.x() + new_dx, fly_cam.eye.y(), target.z() + new_dz);
            fly_cam.yaw += d_angle_deg;
            cam_moved = true;
        }

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
                } else if (e.key.key == SDLK_U) {
                    aperture = std::max(0.0, aperture - 0.02);
                    std::cout << "rt_view: Aperture = " << aperture << " (Focal Dist = " << focus_dist << "m)\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_I) {
                    aperture = std::min(1.0, aperture + 0.02);
                    std::cout << "rt_view: Aperture = " << aperture << " (Focal Dist = " << focus_dist << "m)\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_O) {
                    aperture = (aperture <= 0.001) ? 0.08 : 0.0;
                    std::cout << "rt_view: Aperture toggled to = " << aperture << "\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_K) {
                    focus_dist = std::max(0.1, focus_dist - 0.2);
                    std::cout << "rt_view: Focus Distance = " << focus_dist << "m (Aperture = " << aperture << ")\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_L) {
                    focus_dist = std::min(50.0, focus_dist + 0.2);
                    std::cout << "rt_view: Focus Distance = " << focus_dist << "m (Aperture = " << aperture << ")\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_Z) {
                    temporal_smooth = !temporal_smooth;
                    std::cout << "rt_view: Temporal Motion Smoothing [" << (temporal_smooth ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_B) {
                    blades = (blades == 0) ? 6 : 0;
                    std::cout << "rt_view: Bokeh Iris Shape: [" << (blades >= 3 ? "Hexagonal 6-Blade" : "Circular") << "]\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_N) {
                    live_denoise = !live_denoise;
                    std::cout << "rt_view: Live Bilateral AOV Denoiser [" << (live_denoise ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_V) {
                    optical_vignette = !optical_vignette;
                    std::cout << "rt_view: Optical Vignetting [" << (optical_vignette ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_M) {
                    bloom_enabled = !bloom_enabled;
                    std::cout << "rt_view: Multi-Scale Bloom & Optical Glare [" << (bloom_enabled ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_J) {
                    anamorphic = (anamorphic == 1.0) ? 2.0 : 1.0;
                    std::cout << "rt_view: Anamorphic Squeeze = " << anamorphic << "x ["
                              << (anamorphic > 1.0 ? "2.0x Oval Bokeh" : "1.0x Spherical") << "]\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_Y) {
                    distortion = (distortion == 0.0) ? 0.20 : 0.0;
                    std::cout << "rt_view: Lens Distortion = " << distortion << " ["
                              << (distortion != 0.0 ? "0.20 Barrel" : "0.0 Rectilinear") << "]\n";
                    cam_moved = true;
                } else if (e.key.key == SDLK_9 || e.key.key == SDLK_F5) {
                    chromatic_aberration = !chromatic_aberration;
                    std::cout << "rt_view: Lens Chromatic Aberration ["
                              << (chromatic_aberration ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_F6) {
                    focus_peaking = !focus_peaking;
                    std::cout << "rt_view: Focus Peaking (Z-Peaking) [" << (focus_peaking ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_F8) {
                    turntable_mode = !turntable_mode;
                    std::cout << "rt_view: Cinematic 360 Turntable Orbit [" << (turntable_mode ? "ON" : "OFF") << "]\n";
                } else if (e.key.key == SDLK_LEFT) {
                    double az, el;
                    env_light::get_sun_angles(az, el);
                    az = std::fmod(az - 5.0 + 360.0, 360.0);
                    env_light::set_sun_angles(az, el);
                    cam_moved = true;
                    std::cout << "rt_view: [Sun Orbit] Azimuth = " << az << " deg, Elevation = " << el << " deg\n";
                } else if (e.key.key == SDLK_RIGHT) {
                    double az, el;
                    env_light::get_sun_angles(az, el);
                    az = std::fmod(az + 5.0, 360.0);
                    env_light::set_sun_angles(az, el);
                    cam_moved = true;
                    std::cout << "rt_view: [Sun Orbit] Azimuth = " << az << " deg, Elevation = " << el << " deg\n";
                } else if (e.key.key == SDLK_UP) {
                    double az, el;
                    env_light::get_sun_angles(az, el);
                    el = std::clamp(el + 2.5, 2.0, 88.0);
                    env_light::set_sun_angles(az, el);
                    cam_moved = true;
                    std::cout << "rt_view: [Sun Orbit] Azimuth = " << az << " deg, Elevation = " << el << " deg\n";
                } else if (e.key.key == SDLK_DOWN) {
                    double az, el;
                    env_light::get_sun_angles(az, el);
                    el = std::clamp(el - 2.5, 2.0, 88.0);
                    env_light::set_sun_angles(az, el);
                    cam_moved = true;
                    std::cout << "rt_view: [Sun Orbit] Azimuth = " << az << " deg, Elevation = " << el << " deg\n";
                } else if (e.key.key == SDLK_MINUS) {
                    if (selected_mat) {
                        if (auto m = dynamic_cast<metal*>(selected_mat.get())) {
                            m->set_roughness(m->get_roughness() - 0.05);
                            std::cout << "rt_view: [Live Material Tweaker] Metal Roughness = " << m->get_roughness() << "\n";
                            cam_moved = true;
                        } else if (auto d = dynamic_cast<dielectric*>(selected_mat.get())) {
                            d->set_roughness(d->get_roughness() - 0.05);
                            std::cout << "rt_view: [Live Material Tweaker] Glass Roughness = " << d->get_roughness() << "\n";
                            cam_moved = true;
                        } else if (auto dl = dynamic_cast<diffuse_light*>(selected_mat.get())) {
                            dl->set_emit(dl->get_emit() * 0.8);
                            std::cout << "rt_view: [Live Material Tweaker] Light Emission = ("
                                      << dl->get_emit().x() << ", " << dl->get_emit().y() << ", " << dl->get_emit().z() << ")\n";
                            cam_moved = true;
                        }
                    }
                } else if (e.key.key == SDLK_EQUALS) {
                    if (selected_mat) {
                        if (auto m = dynamic_cast<metal*>(selected_mat.get())) {
                            m->set_roughness(m->get_roughness() + 0.05);
                            std::cout << "rt_view: [Live Material Tweaker] Metal Roughness = " << m->get_roughness() << "\n";
                            cam_moved = true;
                        } else if (auto d = dynamic_cast<dielectric*>(selected_mat.get())) {
                            d->set_roughness(d->get_roughness() + 0.05);
                            std::cout << "rt_view: [Live Material Tweaker] Glass Roughness = " << d->get_roughness() << "\n";
                            cam_moved = true;
                        } else if (auto dl = dynamic_cast<diffuse_light*>(selected_mat.get())) {
                            dl->set_emit(dl->get_emit() * 1.25);
                            std::cout << "rt_view: [Live Material Tweaker] Light Emission = ("
                                      << dl->get_emit().x() << ", " << dl->get_emit().y() << ", " << dl->get_emit().z() << ")\n";
                            cam_moved = true;
                        }
                    }
                } else if (e.key.key == SDLK_COMMA) {
                    if (selected_mat) {
                        if (auto d = dynamic_cast<dielectric*>(selected_mat.get())) {
                            d->set_ior(d->ior() - 0.05);
                            std::cout << "rt_view: [Live Material Tweaker] Glass IOR = " << d->ior() << "\n";
                            cam_moved = true;
                        } else if (auto dl = dynamic_cast<diffuse_light*>(selected_mat.get())) {
                            dl->set_temperature(dl->get_temperature() - 500.0);
                            std::cout << "rt_view: [Live Material Tweaker] Light Temperature = "
                                      << dl->get_temperature() << "K (Warmer) | Emission = ("
                                      << dl->get_emit().x() << ", " << dl->get_emit().y() << ", " << dl->get_emit().z() << ")\n";
                            cam_moved = true;
                        }
                    }
                } else if (e.key.key == SDLK_PERIOD) {
                    if (selected_mat) {
                        if (auto d = dynamic_cast<dielectric*>(selected_mat.get())) {
                            d->set_ior(d->ior() + 0.05);
                            std::cout << "rt_view: [Live Material Tweaker] Glass IOR = " << d->ior() << "\n";
                            cam_moved = true;
                        } else if (auto dl = dynamic_cast<diffuse_light*>(selected_mat.get())) {
                            dl->set_temperature(dl->get_temperature() + 500.0);
                            std::cout << "rt_view: [Live Material Tweaker] Light Temperature = "
                                      << dl->get_temperature() << "K (Cooler) | Emission = ("
                                      << dl->get_emit().x() << ", " << dl->get_emit().y() << ", " << dl->get_emit().z() << ")\n";
                            cam_moved = true;
                        }
                    }
                } else if (e.key.key == SDLK_SEMICOLON) {
                    color_temp = std::max(-0.4, color_temp - 0.05);
                    std::cout << "rt_view: Color Temperature = " << color_temp << " (Cooler)\n";
                } else if (e.key.key == SDLK_APOSTROPHE) {
                    color_temp = std::min(0.4, color_temp + 0.05);
                    std::cout << "rt_view: Color Temperature = " << color_temp << " (Warmer)\n";
                } else if (e.key.key == SDLK_SLASH) {
                    color_temp = 0.0;
                    std::cout << "rt_view: Color Temperature reset to Neutral (0.0)\n";
                } else if (e.key.key == SDLK_F4) {
                    double u = 0.5, v = 0.5;
                    camera probe_cam = fly_cam.build_camera(aspect, 0.0, 1.0);
                    ray probe_r = probe_cam.get_ray(u, v);
                    hit_record probe_rec;
                    if (world.hit(probe_r, 1e-4, 1e30, probe_rec)) {
                        focus_dist = std::max(0.05, probe_rec.t);
                        std::cout << "rt_view: [Autofocus Center] Locked target at dist = " << focus_dist
                                  << " m | Aperture: " << aperture << "\n";
                        cam_moved = true;
                    }
                } else if (e.key.key == SDLK_R) {
                    fly_cam.init_from_camera(sdata.cam, initial_vfov);
                    focus_dist = 4.0;
                    cam_moved = true;
                    std::cout << "rt_view: Camera reset to origin.\n";
                } else if (e.key.key == SDLK_1) {
                    if (scene_name == "studio") {
                        fly_cam.eye = vec3(0.0, 1.15, 1.45);
                        fly_cam.yaw = -90.0; fly_cam.pitch = -4.0; fly_cam.vfov = 34.0;
                        focus_dist = 1.45;
                        std::cout << "rt_view: [Studio Bookmark 1] Centerpiece Cauchy Glass Sphere (dist=" << focus_dist << "m)\n";
                    } else {
                        fly_cam.eye = vec3(0.0, 0.68, 0.35);
                        fly_cam.yaw = -90.0; fly_cam.pitch = -10.0; fly_cam.vfov = 38.0;
                        focus_dist = 1.05;
                        std::cout << "rt_view: [Bookmark 1] Focus on Centerpiece Stanford Bunny (dist=" << focus_dist << "m)\n";
                    }
                    cam_moved = true;
                } else if (e.key.key == SDLK_2) {
                    if (scene_name == "studio") {
                        fly_cam.eye = vec3(-0.95, 0.90, 1.35);
                        fly_cam.yaw = -90.0; fly_cam.pitch = -5.0; fly_cam.vfov = 30.0;
                        focus_dist = 1.0;
                        std::cout << "rt_view: [Studio Bookmark 2] Macro on Ruby Gemstone (dist=" << focus_dist << "m)\n";
                    } else {
                        fly_cam.eye = vec3(0.0, 0.30, 1.35);
                        fly_cam.yaw = -90.0; fly_cam.pitch = -9.0; fly_cam.vfov = 44.0;
                        focus_dist = 1.45;
                        std::cout << "rt_view: [Bookmark 2] Focus on Soap Bubble & Cauchy Crystal Pair (dist=" << focus_dist << "m)\n";
                    }
                    cam_moved = true;
                } else if (e.key.key == SDLK_3) {
                    if (scene_name == "studio") {
                        fly_cam.eye = vec3(1.2, 0.85, 1.7);
                        fly_cam.yaw = -125.0; fly_cam.pitch = -8.0; fly_cam.vfov = 40.0;
                        focus_dist = 1.6;
                        std::cout << "rt_view: [Studio Bookmark 3] Pedestal Gold Ring & Columns (dist=" << focus_dist << "m)\n";
                    } else {
                        fly_cam.eye = vec3(0.0, 0.15, 1.85);
                        fly_cam.yaw = -90.0; fly_cam.pitch = -6.0; fly_cam.vfov = 52.0;
                        focus_dist = 2.30;
                        std::cout << "rt_view: [Bookmark 3] Focus on Disney Car Paint & Velvet Flanks (dist=" << focus_dist << "m)\n";
                    }
                    cam_moved = true;
                } else if (e.key.key == SDLK_4) {
                    fly_cam.init_from_camera(sdata.cam, initial_vfov);
                    focus_dist = (scene_name == "studio") ? 3.2 : 4.0;
                    cam_moved = true;
                    std::cout << "rt_view: [Bookmark 4] Wide Studio Overview (dist=" << focus_dist << "m)\n";
                } else if (e.key.key == SDLK_F1) {
                    view_mode = ViewMode::BEAUTY;
                    std::cout << "rt_view: Display mode [F1: Beauty]\n";
                } else if (e.key.key == SDLK_F2) {
                    view_mode = ViewMode::ALBEDO;
                    std::cout << "rt_view: Display mode [F2: Albedo AOV Guide]\n";
                } else if (e.key.key == SDLK_F3) {
                    view_mode = ViewMode::NORMAL;
                    std::cout << "rt_view: Display mode [F3: Normal AOV Guide]\n";
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
                              << " Mid Click/F4 : Click-to-Focus / Center Autofocus (sets focal plane)\n"
                              << " U / I / O    : Adjust Aperture (U: -0.02, I: +0.02, O: Toggle pinhole/bokeh)\n"
                              << " K / L        : Adjust Focus Distance (K: -0.2m, L: +0.2m)\n"
                              << " B            : Toggle Bokeh Iris Shape (Hexagonal 6-Blade vs Circular)\n"
                              << " J            : Toggle Anamorphic Lens Squeeze (1.0x vs 2.0x Oval Bokeh)\n"
                              << " Y            : Toggle Lens Radial Distortion (0.0 Rectilinear vs 0.20 Barrel)\n"
                              << " 9 / F5       : Toggle Lens Chromatic Aberration & Spectral Fringe\n"
                              << " F6           : Toggle Focus Peaking / Cinema Z-Peaking Overlay\n"
                              << " F8           : Toggle Cinematic 360 Turntable Orbit\n"
                              << " Arrow Keys   : Dynamic Sun Orbit (Left/Right: Azimuth, Up/Down: Elevation)\n"
                              << " - / =        : Nudge Roughness (-0.05 / +0.05) or Light Emission (0.8x / 1.25x)\n"
                              << " , / .        : Nudge Glass IOR (-0.05 / +0.05) or Light Temperature (-500K / +500K)\n"
                              << " N            : Toggle Live Bilateral AOV Denoising\n"
                              << " V            : Toggle Optical Vignetting\n"
                              << " M            : Toggle Multi-Scale Bloom & Optical Glare\n"
                              << " ; / ' / /    : Color Temperature (Cooler / Warmer / Reset to 0.0)\n"
                              << " Z            : Toggle Temporal Motion Smoothing [ON/OFF]\n"
                              << " 1 - 4        : Camera bookmarks (Bunny / Crystals / Disney / Wide)\n"
                              << " F1 - F3      : Display Mode (F1: Beauty, F2: Albedo AOV, F3: Normal AOV)\n"
                              << " X            : Toggle Accumulation Pause / Resume\n"
                              << " G            : Toggle GPU compute vs CPU multi-threading\n"
                              << " T            : Toggle ACES film tonemapping vs Linear/sRGB\n"
                              << " [ / ]        : Exposure decrease / increase\n"
                              << " R            : Reset camera to origin\n"
                              << " P / F12      : Save viewport snapshot to out/viewport.ppm\n"
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
                    // Flip bottom-first for standard PPM
                    std::vector<vec3> ppm_fb((size_t)W * H);
                    double inv = (accum_spp > 0) ? (1.0 / accum_spp) : 1.0;
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x)
                            ppm_fb[((size_t)H - 1 - y) * W + x] = accum_fb[(size_t)y * W + x] * inv;
                    if (write_ppm("out/viewport.ppm", ppm_fb, W, H, exposure))
                        std::cout << "rt_view: Saved current viewport to out/viewport.ppm\n\n";
                }
            } else if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (e.button.button == SDL_BUTTON_RIGHT) {
                    right_mouse_down = true;
                    SDL_SetWindowRelativeMouseMode(win, true);
                } else if (e.button.button == SDL_BUTTON_MIDDLE ||
                           (e.button.button == SDL_BUTTON_LEFT && (SDL_GetModState() & SDL_KMOD_CTRL))) {
                    int mx = std::clamp((int)e.button.x / scale, 0, W - 1);
                    int my = std::clamp((int)e.button.y / scale, 0, H - 1);
                    double u = (mx + 0.5) / (double)W;
                    double v = (H - 1 - my + 0.5) / (double)H;
                    camera probe_cam = fly_cam.build_camera(aspect, 0.0, 1.0);
                    ray probe_r = probe_cam.get_ray(u, v);
                    hit_record probe_rec;
                    if (world.hit(probe_r, 1e-4, 1e30, probe_rec)) {
                        focus_dist = std::max(0.05, probe_rec.t);
                        std::cout << "rt_view: [Click-to-Focus] Locked target at (" << mx << ", " << my
                                  << ") -> dist = " << focus_dist << " m | Aperture: " << aperture << "\n";
                        cam_moved = true;
                    } else {
                        std::cout << "rt_view: [Click-to-Focus] Missed geometry (infinity)\n";
                    }
                } else if (e.button.button == SDL_BUTTON_LEFT && (SDL_GetModState() & SDL_KMOD_ALT)) {
                    int mx = std::clamp((int)e.button.x / scale, 0, W - 1);
                    int my = std::clamp((int)e.button.y / scale, 0, H - 1);
                    double u = (mx + 0.5) / (double)W;
                    double v = (H - 1 - my + 0.5) / (double)H;
                    camera probe_cam = fly_cam.build_camera(aspect, 0.0, 1.0);
                    ray probe_r = probe_cam.get_ray(u, v);
                    hit_record probe_rec;
                    if (world.hit(probe_r, 1e-4, 1e30, probe_rec)) {
                        selected_mat = probe_rec.mat;
                        std::cout << "\n=== [Object Inspector] ===\n"
                                  << "  Screen Pixel   : (" << mx << ", " << my << ")\n"
                                  << "  Hit Distance   : " << probe_rec.t << " m\n"
                                  << "  World Position : (" << probe_rec.point.x() << ", "
                                  << probe_rec.point.y() << ", " << probe_rec.point.z() << ")\n"
                                  << "  Shading Normal : (" << probe_rec.normal.x() << ", "
                                  << probe_rec.normal.y() << ", " << probe_rec.normal.z() << ")\n"
                                  << "  Geo Normal     : (" << probe_rec.geo_normal.x() << ", "
                                  << probe_rec.geo_normal.y() << ", " << probe_rec.geo_normal.z() << ")\n"
                                  << "  Surface UV     : (" << probe_rec.u << ", " << probe_rec.v << ")\n"
                                  << "  Primitive Ptr  : " << probe_rec.hit_prim << "\n"
                                  << "  Material Ptr   : " << probe_rec.mat.get() << "\n"
                                  << "  --> Selected for Live Editing: [-/=] Roughness, [, / .] IOR\n\n";
                    } else {
                        std::cout << "rt_view: [Object Inspector] Missed geometry (background / sky)\n";
                    }
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

        // Reset accumulation on camera movement
        if (cam_moved) {
            accum_spp = 0;
            std::fill(accum_fb.begin(), accum_fb.end(), vec3(0, 0, 0));
            active_cam = fly_cam.build_camera(aspect, aperture, focus_dist, blades, anamorphic, distortion);
            if (gpu.has_scene) {
                GPUCam gcam = gpu_cam_from_cpu(active_cam.eye(), active_cam.corner(),
                                               active_cam.span_u(), active_cam.span_v(),
                                               active_cam.lens_r(), blades,
                                               anamorphic, distortion);
                gpu_update_camera(gpu, &gcam, sizeof(gcam));
            }
        }

        // Render pass: dispatch progressive path tracing when not paused
        if (!accum_paused) {
            if (use_gpu) {
                push.spp = 1;
                push.seed = (uint32_t)accum_spp + 1;
                gpu_run(gpu, shader_path, push, gpu_rgba);
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    accum_fb[i] += vec3(gpu_rgba[i * 4 + 0], gpu_rgba[i * 4 + 1], gpu_rgba[i * 4 + 2]);
                }
                accum_spp += 1;
            } else {
                // Multi-threaded CPU progressive slice
                int chunk_h = (H + (int)num_threads - 1) / (int)num_threads;
                std::vector<std::thread> workers;
                workers.reserve(num_threads);
                for (unsigned t = 0; t < num_threads; ++t) {
                    int y_start = (int)t * chunk_h;
                    int y_end = std::min(H, y_start + chunk_h);
                    if (y_start >= y_end) continue;
                    workers.emplace_back([&, y_start, y_end, t] {
                        rng_seed(42u + (unsigned)accum_spp * 10007u + t * 997u);
                        for (int sy = y_start; sy < y_end; ++sy) {
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
                }
                for (auto &w : workers)
                    w.join();
                accum_spp += 1;
            }
        }

        // Convert active view mode to display RGB24
        if (view_mode == ViewMode::BEAUTY) {
            double inv_spp = (accum_spp > 0) ? (1.0 / (double)accum_spp) : 1.0;
            std::vector<vec3> beauty_hdr((size_t)W * H);
            for (size_t i = 0; i < (size_t)W * H; ++i)
                beauty_hdr[i] = accum_fb[i] * inv_spp;
            if (live_denoise && accum_spp >= 2)
                beauty_hdr = bilateral_denoise(beauty_hdr, W, H, 1.5, 0.15);
            if (bloom_enabled)
                beauty_hdr = bloom::apply_bloom(beauty_hdr, W, H, 1.0, 0.08);
            double half_w = W * 0.5, half_h = H * 0.5;
            for (int y = 0; y < H; ++y) {
                double dy = (y - half_h) / half_h;
                for (int x = 0; x < W; ++x) {
                    double dx = (x - half_w) / half_w;
                    size_t i = (size_t)y * W + x;
                    vec3 hdr = beauty_hdr[i];
                    if (chromatic_aberration) {
                        double r2 = dx * dx + dy * dy;
                        int offset_x = (int)std::round(dx * r2 * 4.0);
                        int offset_y = (int)std::round(dy * r2 * 4.0);
                        int rx = std::clamp(x + offset_x, 0, W - 1);
                        int ry = std::clamp(y + offset_y, 0, H - 1);
                        int bx = std::clamp(x - offset_x, 0, W - 1);
                        int by = std::clamp(y - offset_y, 0, H - 1);
                        hdr = vec3(beauty_hdr[(size_t)ry * W + rx].x(),
                                   hdr.y(),
                                   beauty_hdr[(size_t)by * W + bx].z());
                    }
                    if (color_temp != 0.0)
                        hdr = vec3(hdr.x() * (1.0 + color_temp), hdr.y(), hdr.z() * (1.0 - color_temp));
                    if (optical_vignette) {
                        hdr *= 1.0 / (1.0 + 0.45 * (dx * dx + dy * dy));
                    }
                    vec3 ldr = use_aces ? tonemap(hdr, exposure)
                                        : vec3(srgb_encode(hdr.x() * exposure),
                                               srgb_encode(hdr.y() * exposure),
                                               srgb_encode(hdr.z() * exposure));
                    if (focus_peaking) {
                        auto get_lum = [&](int px, int py) {
                            px = std::clamp(px, 0, W - 1);
                            py = std::clamp(py, 0, H - 1);
                            vec3 c = beauty_hdr[(size_t)py * W + px];
                            return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
                        };
                        double y_c = get_lum(x, y);
                        double y_l = get_lum(x - 1, y);
                        double y_r = get_lum(x + 1, y);
                        double y_u = get_lum(x, y - 1);
                        double y_d = get_lum(x, y + 1);
                        double lap = std::abs(4.0 * y_c - y_l - y_r - y_u - y_d) / (y_c + 0.05);
                        if (lap > 0.22) {
                            ldr = 0.25 * ldr + 0.75 * vec3(0.0, 1.0, 0.4);
                        }
                    }
                    uint8_t r = (uint8_t)(std::clamp(ldr.x(), 0.0, 1.0) * 255.999);
                    uint8_t g = (uint8_t)(std::clamp(ldr.y(), 0.0, 1.0) * 255.999);
                    uint8_t b = (uint8_t)(std::clamp(ldr.z(), 0.0, 1.0) * 255.999);
                    if (temporal_smooth && accum_spp <= 2 && prev_display_rgb.size() == display_rgb.size() && prev_display_rgb[i * 3 + 0] != 0) {
                        display_rgb[i * 3 + 0] = (uint8_t)(0.40f * r + 0.60f * prev_display_rgb[i * 3 + 0]);
                        display_rgb[i * 3 + 1] = (uint8_t)(0.40f * g + 0.60f * prev_display_rgb[i * 3 + 1]);
                        display_rgb[i * 3 + 2] = (uint8_t)(0.40f * b + 0.60f * prev_display_rgb[i * 3 + 2]);
                    } else {
                        display_rgb[i * 3 + 0] = r;
                        display_rgb[i * 3 + 1] = g;
                        display_rgb[i * 3 + 2] = b;
                    }
                }
            }
            prev_display_rgb = display_rgb;
        } else if (view_mode == ViewMode::ALBEDO) {
            if (use_gpu && gpu.has_scene) {
                gpu_read_aov(gpu, aov_alb, aov_nrm);
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    display_rgb[i * 3 + 0] = (uint8_t)(std::clamp(srgb_encode(aov_alb[i * 4 + 0]), 0.0, 1.0) * 255.999);
                    display_rgb[i * 3 + 1] = (uint8_t)(std::clamp(srgb_encode(aov_alb[i * 4 + 1]), 0.0, 1.0) * 255.999);
                    display_rgb[i * 3 + 2] = (uint8_t)(std::clamp(srgb_encode(aov_alb[i * 4 + 2]), 0.0, 1.0) * 255.999);
                }
            } else {
                for (int y = 0; y < H; ++y) {
                    int j = H - 1 - y;
                    for (int x = 0; x < W; ++x) {
                        double u = (x + 0.5) / (double)W;
                        double v = (j + 0.5) / (double)H;
                        ray r = active_cam.get_ray(u, v);
                        vec3 a, n;
                        bool hit;
                        first_hit_aov(r, world, a, n, hit);
                        size_t idx = (size_t)y * W + x;
                        display_rgb[idx * 3 + 0] = (uint8_t)(std::clamp(srgb_encode(a.x()), 0.0, 1.0) * 255.999);
                        display_rgb[idx * 3 + 1] = (uint8_t)(std::clamp(srgb_encode(a.y()), 0.0, 1.0) * 255.999);
                        display_rgb[idx * 3 + 2] = (uint8_t)(std::clamp(srgb_encode(a.z()), 0.0, 1.0) * 255.999);
                    }
                }
            }
        } else if (view_mode == ViewMode::NORMAL) {
            if (use_gpu && gpu.has_scene) {
                gpu_read_aov(gpu, aov_alb, aov_nrm);
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    display_rgb[i * 3 + 0] = (uint8_t)(std::clamp(0.5f * aov_nrm[i * 4 + 0] + 0.5f, 0.0f, 1.0f) * 255.999f);
                    display_rgb[i * 3 + 1] = (uint8_t)(std::clamp(0.5f * aov_nrm[i * 4 + 1] + 0.5f, 0.0f, 1.0f) * 255.999f);
                    display_rgb[i * 3 + 2] = (uint8_t)(std::clamp(0.5f * aov_nrm[i * 4 + 2] + 0.5f, 0.0f, 1.0f) * 255.999f);
                }
            } else {
                for (int y = 0; y < H; ++y) {
                    int j = H - 1 - y;
                    for (int x = 0; x < W; ++x) {
                        double u = (x + 0.5) / (double)W;
                        double v = (j + 0.5) / (double)H;
                        ray r = active_cam.get_ray(u, v);
                        vec3 a, n;
                        bool hit;
                        first_hit_aov(r, world, a, n, hit);
                        size_t idx = (size_t)y * W + x;
                        display_rgb[idx * 3 + 0] = (uint8_t)(std::clamp(0.5 * n.x() + 0.5, 0.0, 1.0) * 255.999);
                        display_rgb[idx * 3 + 1] = (uint8_t)(std::clamp(0.5 * n.y() + 0.5, 0.0, 1.0) * 255.999);
                        display_rgb[idx * 3 + 2] = (uint8_t)(std::clamp(0.5 * n.z() + 0.5, 0.0, 1.0) * 255.999);
                    }
                }
            }
        }

        SDL_UpdateTexture(tex, nullptr, display_rgb.data(), W * 3);
        SDL_RenderClear(ren);
        SDL_RenderTexture(ren, tex, nullptr, nullptr);
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

            const char *mode_str = (view_mode == ViewMode::BEAUTY) ? "Beauty" :
                                   (view_mode == ViewMode::ALBEDO) ? "Albedo AOV" : "Normal AOV";
            std::snprintf(title_buf, sizeof(title_buf),
                          "rt_view [%s] %s | %s%s | SPP: %d | %.1f FPS (%.1f ms) | Ap: %.2f | Foc: %.2fm | TS: %s | Bokeh: %s | Den: %s | Vig: %s | Bloom: %s | Chr: %s | Anam: %.1fx",
                          use_gpu ? "GPU" : "CPU", scene_name.c_str(), mode_str,
                          accum_paused ? " [PAUSED]" : "", accum_spp,
                          current_fps, current_ms, aperture, focus_dist,
                          temporal_smooth ? "ON" : "OFF",
                          (blades >= 3 ? "Hex" : "Circ"),
                          (live_denoise ? "ON" : "OFF"),
                          (optical_vignette ? "ON" : "OFF"),
                          (bloom_enabled ? "ON" : "OFF"),
                          (chromatic_aberration ? "ON" : "OFF"),
                          anamorphic);
            SDL_SetWindowTitle(win, title_buf);
        }
    }

    if (gpu.has_scene) {
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
