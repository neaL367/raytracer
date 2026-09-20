// rt_gpu: headless Vulkan compute backend. Thin wiring only:
// args -> scene_data -> flat BVH -> dispatch -> PPM. Mechanics in
// vk_compute.h, numbers in scene/scene.h via flatten.h.
// Usage: rt_gpu [shader.spv] [out.ppm] [--spp N] [--seed S]
//        [--scene NAME] [--width W] [--height H]
#include "host_scene.h"
#include "flatten.h"
#include "vk_compute.h"
#include "output/ppm.h"
#include "scene/scene.h"

#include "core/vec3.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    double shutter0 = 0, shutter1 = 0, fog_density = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc)
            spp = std::max(1, std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (a == "--scene" && i + 1 < argc)
            scene_name = argv[++i];
        else if (a == "--shutter" && i + 2 < argc) {
            shutter0 = std::atof(argv[++i]);
            shutter1 = std::atof(argv[++i]);
        } else if (a == "--fog" && i + 1 < argc)
            fog_density = std::max(0.0, std::atof(argv[++i]));
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
        H = (int)((double)W / (16.0 / 9.0)); // same default as CPU app
    auto t0 = std::chrono::high_resolution_clock::now();

    // One construction order with the CPU (scene/scene.h): flatten the
    // same scene to typed arrays + BVH nodes instead of hardcoded data.
    scene_data sdata =
        build_scene(scene_name, (double)W / (double)H, 0.0, shutter0, shutter1, fog_density);
    flat_scene flat;
    if (!flatten_scene(sdata, flat)) {
        std::cerr << "scene has non-exportable shapes\n";
        return 1;
    }
    gpu_scene &scene = flat.gs;
    scene.cam = build_gpu_camera(W, H, scene_name);
    // Image table: (offset, w, h, levels) rows over the concatenated blob,
    // each followed by a (spanbits, 0, 0, 0) row for distance LOD.
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
    GpuContext gpu{};
    gpu_init(gpu, W, H);
    // Image texture blob: empty (16B pad) when the scene has none.
    static const float empty_blob[4] = {};
    const void *img_ptr = img_blob.empty() ? empty_blob : img_blob.data();
    size_t img_bytes = img_blob.empty() ? sizeof empty_blob : img_blob.size() * sizeof(float);
    static const int empty_tab[4] = {};
    const void *tab_ptr = img_table.empty() ? empty_tab : img_table.data();
    size_t tab_bytes = img_table.empty() ? sizeof empty_tab : img_table.size() * sizeof(int);
    const void *data[8] = {&scene.cam, scene.spheres.data(), scene.quads.data(),
                           scene.tris.data(), flat.nodes.data(), flat.refs.data(),
                           img_ptr, tab_ptr};
    const size_t bytes[8] = {sizeof scene.cam, scene.spheres.size() * sizeof(GPUSphere),
                             scene.quads.size() * sizeof(GPUQuad),
                             scene.tris.size() * sizeof(GPUTri),
                             flat.nodes.size() * sizeof(GPUNode),
                             flat.refs.size() * sizeof(GPURef), img_bytes, tab_bytes};
    gpu_set_scene(gpu, data, bytes);

    uint32_t push10[10] = {(uint32_t)W,         (uint32_t)H,
                           (uint32_t)scene.spheres.size(),
                           (uint32_t)scene.quads.size(), (uint32_t)scene.tris.size(),
                           (uint32_t)spp,                (uint32_t)seed,
                           (uint32_t)flat.nlights,       0, 0};
    float shf[2] = {(float)shutter0, (float)shutter1};
    std::memcpy(&push10[8], shf, sizeof shf);
    // Fog-slot count gates fog RNG draws (static streams bit-exact).
    int nfog = 0;
    for (const auto &s : scene.spheres)
        if (s.prm[0] == 6)
            nfog++;
    uint32_t push11[11];
    for (int k = 0; k < 10; ++k)
        push11[k] = push10[k];
    push11[10] = (uint32_t)nfog;
    std::vector<float> rgba;
    double dispatch_ms = gpu_run(gpu, shader, push11, rgba);

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
