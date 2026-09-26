// rt_gpu: headless Vulkan compute backend. Thin wiring only:
// args -> scene_data -> flat BVH -> dispatch -> PPM. Mechanics in
// vk_compute.h, numbers in scene/scene.h via flatten.h.
// Usage: rt_gpu [shader.spv] [out.ppm] [--spp N] [--seed S]
//        [--scene NAME] [--width W] [--height H] [--hdr float.pfm] [--denoise]
//        [--joint]
//        [--shutter T0 T1] [--fog D] [--het D] [--noise] [--env]
//        [--aperture A] [--exposure X]
#include "host_scene.h"
#include "flatten.h"
#include "vk_compute.h"
#include "output/ppm.h"
#include "output/pfm.h"
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
    bool spp_set = false, width_set = false;
    int max_depth = 50; // bounce cap (depth ladder forensics)
    bool fixed_rng = false; // --fixed-rng: deterministic 0.5 stream (M54)
    int chunk_spp = 0; // --chunk C: spp per dispatch (TDR); 0 = one shot
    std::string scene_name = "default";
    std::string hdr_path; // empty = no float dump
    bool do_denoise = false;
    bool do_joint = false;
    bool dump_aov = false; // --aov: download albedo/normal guides as PFM
    double aperture = 0.0, exposure = 1.0;
    double shutter0 = 0, shutter1 = 0, fog_density = 0, het_density = 0;
    bool marble_demo = false, env_demo = false;
    std::string hdri_env_path; // --hdri-env <file.hdr>
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc) {
            spp = std::max(1, std::atoi(argv[++i]));
            spp_set = true;
        }
        else if (a == "--chunk" && i + 1 < argc)
            chunk_spp = std::max(0, std::atoi(argv[++i]));
        else if (a == "--seed" && i + 1 < argc)
            seed = std::atoi(argv[++i]);
        else if (a == "--scene" && i + 1 < argc)
            scene_name = argv[++i];
        else if (a == "--list-scenes") {
            print_scenes();
            return 0;
        }
        else if (a == "--shutter" && i + 2 < argc) {
            shutter0 = std::atof(argv[++i]);
            shutter1 = std::atof(argv[++i]);
        } else if (a == "--fog" && i + 1 < argc)
            fog_density = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--het" && i + 1 < argc)
            het_density = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--noise")
            marble_demo = true;
        else if (a == "--env")
            env_demo = true;
        else if (a == "--hdri-env" && i + 1 < argc)
            hdri_env_path = argv[++i];
        else if (a == "--width" && i + 1 < argc) {
            W = std::max(8, std::atoi(argv[++i]));
            width_set = true;
        }
        else if (a == "--height" && i + 1 < argc)
            H = std::max(8, std::atoi(argv[++i]));
        else if (a == "--hdr" && i + 1 < argc)
            hdr_path = argv[++i];
        else if (a == "--denoise")
            do_denoise = true;
        else if (a == "--joint")
            do_joint = true;
        else if (a == "--aov")
            dump_aov = true;
        else if (a == "--aperture" && i + 1 < argc)
            aperture = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--exposure" && i + 1 < argc)
            exposure = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--maxdepth" && i + 1 < argc)
            max_depth = std::max(1, std::atoi(argv[++i]));
        else if (a == "--fixed-rng")
            fixed_rng = true;
        else if (a.ends_with(".spv"))
            shader = a;
        else if (a.ends_with(".ppm"))
            out_path = a;
    }
    bool is_showcase = (scene_name == "showcase" || scene_name == "show" ||
                        scene_name == "demo");
    if (is_showcase) {
        if (!width_set)
            W = 1200;
        if (H <= 0)
            H = 675;
        if (!spp_set)
            spp = 512;
        if (chunk_spp == 0)
            chunk_spp = 64;
    }
    if (H <= 0)
        H = (int)((double)W / (16.0 / 9.0)); // same default as CPU app
    auto t0 = std::chrono::high_resolution_clock::now();

    // One construction order with the CPU (scene/scene.h): flatten the
    // same scene to typed arrays + BVH nodes instead of hardcoded data.
    scene_data sdata =
        build_scene(scene_name, (double)W / (double)H, 0.0, shutter0, shutter1, fog_density,
                    het_density, marble_demo, env_demo);
    // M63: load HDRI if requested.
    if (!hdri_env_path.empty()) {
        sdata.hdri = std::make_shared<hdri_env>();
        if (!sdata.hdri->load(hdri_env_path)) {
            std::cerr << "hdri-env: could not load '" << hdri_env_path << "'\n";
            sdata.hdri.reset();
        } else {
            sdata.env_light = true; // activate env MIS path (nenv_mode()==2)
            sdata.black_bg = false; // HDRI replaces the studio-void background
        }
    }
    flat_scene flat;
    if (!flatten_scene(sdata, flat)) {
        std::cerr << "scene has non-exportable shapes\n";
        return 1;
    }
    gpu_scene &scene = flat.gs;
    // Mirror the CPU camera exactly (M53): hardcoded per-scene cams drifted
    // out of sync with scene tunes (default vfov 90 vs tuned 75).
    scene.cam = gpu_cam_from_cpu(sdata.cam.eye(), sdata.cam.corner(),
                                 sdata.cam.span_u(), sdata.cam.span_v(),
                                 sdata.cam.lens_r());
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
    // NEE light table: (type, index) ivec2 rows; empty pad when none.
    std::vector<int> light_tab;
    for (const auto &le : flat.light_table) {
        light_tab.push_back(le.first);
        light_tab.push_back(le.second);
    }
    static const int empty_light[2] = {};
    const void *light_ptr = light_tab.empty() ? empty_light : light_tab.data();
    size_t light_bytes =
        light_tab.empty() ? sizeof empty_light : light_tab.size() * sizeof(int);
    // M63: HDRI texel + CDF buffers (empty pads when no HDRI loaded).
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
    // Light power CDF over the GPU light TABLE (per-tri entries for
    // tessellated shapes): weights mirror the shader densities, so the
    // GPU estimator stays internally consistent (see flatten.h).
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

    const void *data[12] = {&scene.cam, scene.spheres.data(), scene.quads.data(),
                            scene.tris.data(), flat.nodes.data(), flat.refs.data(),
                            img_ptr, tab_ptr, light_ptr, hdri_tex_ptr, hdri_cdf_ptr,
                            light_cdf_ptr};
    const size_t bytes[12] = {sizeof scene.cam, scene.spheres.size() * sizeof(GPUSphere),
                              scene.quads.size() * sizeof(GPUQuad),
                              scene.tris.size() * sizeof(GPUTri),
                              flat.nodes.size() * sizeof(GPUQNode),
                              flat.refs.size() * sizeof(GPURef), img_bytes, tab_bytes,
                              light_bytes, hdri_tex_bytes, hdri_cdf_bytes,
                              light_cdf_bytes};
    gpu_set_scene(gpu, data, bytes);

    // Fog-slot count gates fog RNG draws (static streams bit-exact).
    int nfog = 0;
    for (const auto &s : scene.spheres)
        if (s.prm[0] == static_cast<float>(MatType::FOG) || s.prm[0] == static_cast<float>(MatType::HET))
            nfog++;

    PushConstants push;
    push.W = W;
    push.H = H;
    push.ns = (int)scene.spheres.size();
    push.nq = (int)scene.quads.size();
    push.nt = (int)scene.tris.size();
    push.spp = spp;
    push.seed = seed;
    push.nlights = (int)flat.nlights;
    push.sh0 = (float)shutter0;
    push.sh1 = (float)shutter1;
    push.nfog = nfog;
    push.nenv = sdata.nenv_mode();
    push.nblack = sdata.black_bg ? 1 : 0;
    push.maxdepth = max_depth;
    push.fixed_rng = fixed_rng ? 1 : 0;

    // Chunked submit (M52): split spp into TDR-safe dispatches, accumulate
    // linear HDR on the host in fp64 (same order as the old python script:
    // v[i]/n added per chunk, so chunked output bit-matches manual runs).
    // Chunk k uses seed+k. Chunk 0/absent == legacy single dispatch.
    int per = (chunk_spp > 0) ? chunk_spp : spp;
    int nchunks = (spp + per - 1) / per;
    // Present mode (on-device accum + film, byte out) is automatic for
    // plain LDR renders; post-passes/HDR/AOV stay on the host path.
    bool use_present = !do_denoise && !do_joint && !dump_aov && hdr_path.empty();
    std::string shdir = shader.substr(0, shader.find_last_of("/\\") + 1);
    std::vector<double> acc((size_t)W * H * 4, 0.0);
    std::vector<float> rgba;
    std::vector<unsigned char> present_bytes;
    PresentAccum pacc{};
    double dispatch_ms = 0, present_ms = 0;
    int done = 0;
    for (int c = 0; c < nchunks; ++c) {
        int cspp = std::min(per, spp - done);
        push.spp = cspp;
        push.seed = seed + c;
        if (use_present) {
            // Zero float downloads: chunk stays on device, folded into the
            // persistent fp32 sum (same equal-chunk weighting as host: /n).
            dispatch_ms += gpu_run(gpu, shader, push, rgba, true);
            present_ms += present_accum_add(gpu, pacc, shdir + "accum.spv", c == 0);
        } else {
            dispatch_ms += gpu_run(gpu, shader, push, rgba);
            // NOTE: divide (not multiply-by-reciprocal) to bit-match the old
            // python averaging (a/n per chunk, same order).
            for (size_t k = 0; k < acc.size(); ++k)
                acc[k] += (double)rgba[k] / (double)nchunks;
        }
        done += cspp;
    }
    if (use_present) {
        // Film sum/nchunks on device -> top-first RGBA bytes (P6-ready).
        present_ms += present_tonemap(gpu, pacc, shdir + "tonemap.spv",
                                      (float)exposure, 1, (float)nchunks,
                                      present_bytes);
        present_accum_destroy(gpu, pacc);
    }
    // Post-passes consume the averaged beauty (single dispatch feeds its
    // own output through the same path, unchanged).
    rgba.resize(acc.size());
    for (size_t k = 0; k < acc.size(); ++k)
        rgba[k] = (float)acc[k];
    double denoise_ms = 0;
    if (do_denoise) {
        // Bilateral post-pass on device (linear HDR); needs the denoise
        // shader next to the path shader (same SHADER_DIR).
        std::string dsh = shader.substr(0, shader.find_last_of("/\\") + 1) + "denoise.spv";
        std::vector<float> smooth;
        denoise_ms = gpu_denoise(gpu, dsh, rgba, smooth);
        rgba = std::move(smooth);
    }
    double joint_ms = 0;
    if (do_joint) {
        // Guided pass reads beauty + AOVs on device (no upload); needs
        // joint.spv next to the path shader.
        std::string jsh = shader.substr(0, shader.find_last_of("/\\") + 1) + "joint.spv";
        std::vector<float> guided;
        joint_ms = gpu_joint(gpu, jsh, guided);
        rgba = std::move(guided);
    }

    // Image rows top-first -> flip for PPM writer (bottom-first).
    // Film from the fp64 average directly (not the float round-trip):
    // matches the old python script bit-exactly. Single dispatch feeds
    // identical doubles (float->double is exact), so legacy output is
    // unchanged. Post-passes replace rgba, so film those from rgba.
    bool post = do_denoise || do_joint;
    std::vector<vec3> fb((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            vec3 c;
            if (post) {
                const float *t = rgba.data() + ((size_t)y * W + x) * 4;
                c = vec3(t[0], t[1], t[2]);
            } else {
                const double *t = acc.data() + ((size_t)y * W + x) * 4;
                c = vec3(t[0], t[1], t[2]);
            }
            fb[((size_t)H - 1 - y) * W + x] = c;
    }
    if (dump_aov && !post) {
        // Guide readback mirrors the CPU --aov trio (linear, no film).
        // Chunked runs leave the LAST chunk's guides on device.
        std::vector<float> alb_rgba, nrm_rgba;
        gpu_read_aov(gpu, alb_rgba, nrm_rgba);
        std::vector<vec3> alb_fb((size_t)W * H), nrm_fb((size_t)W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float *ta = alb_rgba.data() + ((size_t)y * W + x) * 4;
                const float *tn = nrm_rgba.data() + ((size_t)y * W + x) * 4;
                alb_fb[((size_t)H - 1 - y) * W + x] = vec3(ta[0], ta[1], ta[2]);
                nrm_fb[((size_t)H - 1 - y) * W + x] = vec3(tn[0], tn[1], tn[2]);
            }
        write_pfm("out/aov_albedo.pfm", alb_fb, W, H);
        write_pfm("out/aov_normal.pfm", nrm_fb, W, H);
    }
    bool wrote_ok = false;
    if (use_present) {
        // Bytes are top-first RGBA from the device; emit P6 directly.
        std::string buf;
        buf.reserve((size_t)W * H * 3 + 32);
        buf.append("P6\n");
        buf.append(std::to_string(W));
        buf.push_back(' ');
        buf.append(std::to_string(H));
        buf.append("\n255\n");
        for (size_t i = 0; i < (size_t)W * H; ++i) {
            buf.push_back((char)present_bytes[i * 4 + 0]);
            buf.push_back((char)present_bytes[i * 4 + 1]);
            buf.push_back((char)present_bytes[i * 4 + 2]);
        }
        std::filesystem::create_directories("out");
        std::ofstream out(out_path, std::ios::binary);
        if (out) {
            out.write(buf.data(), (std::streamsize)buf.size());
            wrote_ok = (bool)out;
        }
    }
    gpu_shutdown(gpu);

    std::filesystem::create_directories("out");
    if (use_present) {
        if (!wrote_ok) {
            std::cerr << "write failed\n";
            return 1;
        }
    } else if (!write_ppm(out_path.c_str(), fb, W, H, exposure)) {
        std::cerr << "write failed\n";
        return 1;
    }
    if (!hdr_path.empty() && !write_pfm(hdr_path.c_str(), fb, W, H)) {
        std::cerr << "hdr dump failed\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "wrote " << out_path << " spp=" << spp
              << " dispatch=" << dispatch_ms << "ms wall="
              << std::chrono::duration<double>(t1 - t0).count() << "s";
    if (use_present)
        std::cout << " present=" << present_ms << "ms";
    if (nchunks > 1)
        std::cout << " chunks=" << nchunks;
    if (do_denoise)
        std::cout << " denoise=" << denoise_ms << "ms";
    if (do_joint)
        std::cout << " joint=" << joint_ms << "ms";
    std::cout << "\n";
    return 0;
}
