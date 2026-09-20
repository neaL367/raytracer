// rt_gpu: headless Vulkan compute backend. Thin wiring only:
// args -> scene bytes -> dispatch -> PPM. Mechanics live in
// vk_compute.h, scene numbers in host_scene.h.
// Usage: rt_gpu [shader.spv] [out.ppm] [--spp N] [--seed S]
#include "host_scene.h"
#include "vk_compute.h"
#include "output/ppm.h"

#include "core/vec3.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    int W = 400, H = -1;
    std::string shader = SHADER_DIR "/grad.spv";
    std::string out_path = "out/gpu_grad.ppm";
    int spp = 16, seed = 42;
    std::string scene_name = "default";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc)
            spp = std::max(1, std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (a == "--scene" && i + 1 < argc)
            scene_name = argv[++i];
        else if (a == "--width" && i + 1 < argc)
            W = std::max(8, std::atoi(argv[++i]));
        else if (a == "--height" && i + 1 < argc)
            H = std::max(8, std::atoi(argv[++i]));
        else if (a.ends_with(".spv"))
            shader = a;
        else if (a.ends_with(".ppm"))
            out_path = a;
    }
    if (H <= 0)
        H = (W * 9 + 8) / 16; // 16:9 default
    auto t0 = std::chrono::high_resolution_clock::now();

    gpu_scene scene = build_gpu_scene(W, H, scene_name);
    GpuContext gpu{};
    gpu_init(gpu, W, H);
    const void *data[4] = {&scene.cam, scene.spheres.data(), scene.quads.data(),
                           scene.tris.data()};
    const size_t bytes[4] = {sizeof scene.cam, scene.spheres.size() * sizeof(GPUSphere),
                             scene.quads.size() * sizeof(GPUQuad),
                             scene.tris.size() * sizeof(GPUTri)};
    gpu_set_scene(gpu, data, bytes);

    int push8[8] = {W, H, (int)scene.spheres.size(), (int)scene.quads.size(),
                    (int)scene.tris.size(), spp, seed, 1};
    std::vector<float> rgba;
    double dispatch_ms = gpu_run(gpu, shader, push8, rgba);

    // Image rows top-first -> flip for PPM writer (bottom-first).
    std::vector<vec3> fb((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float *t = rgba.data() + ((size_t)y * W + x) * 4;
            fb[((size_t)H - 1 - y) * W + x] = vec3(t[0], t[1], t[2]);
        }
    gpu_shutdown(gpu);

    std::filesystem::create_directories("out");
    if (!write_ppm(out_path.c_str(), fb, W, H)) {
        std::cerr << "write failed\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "wrote " << out_path << " spp=" << spp
              << " dispatch=" << dispatch_ms << "ms wall="
              << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    return 0;
}
