// Shading + output tests: materials, textures, emissives, PPM, film.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/texture.h"
#include "core/ggx.h"
#include "geometry/hittable.h"
#include "geometry/quad.h"
#include "material/material.h"
#include "geometry/sphere.h"
#include "integrator/integrator.h"
#include "io/ppm_image.h"
#include "io/stb_loader.h"
#include "io/compare.h"
#include "io/denoise.h"
#include "output/film.h"
#include "output/pfm.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static void t_materials() {
    test_current = "materials";
    rng_seed(4);
    hit_record rec;
    rec.point = vec3(0,0,0); rec.normal = vec3(0,1,0); rec.front_face = true;
    vec3 att; ray sc;
    lambertian lamb(vec3(0.7,0.3,0.3));
    EXPECT_TRUE(lamb.scatter(ray(vec3(0,1,0), vec3(0,-1,0)), rec, att, sc));
    EXPECT_NEAR(att.x(), 0.7);
    EXPECT_TRUE(dot(sc.direction(), rec.normal) > -1.0); // hemisphere-ish
    metal met(vec3(0.8,0.8,0.8), 0.0);
    EXPECT_TRUE(met.scatter(ray(vec3(1,1,0), vec3(-1,-1,0)), rec, att, sc));
    EXPECT_TRUE(sc.direction().x() < 0 && sc.direction().y() > 0); // mirrored
    dielectric glass(1.5);
    rec.front_face = true;
    EXPECT_TRUE(glass.scatter(ray(vec3(0,1,0), vec3(0,-1,0)), rec, att, sc));
    EXPECT_NEAR(att.x(), 1.0); // no absorption
}

static void t_texture() {
    test_current = "texture";
    solid_color solid(vec3(0.2, 0.4, 0.6));
    EXPECT_NEAR(solid.value(0.1, 0.9, vec3(5, 5, 5)).y(), 0.4);
    checker cb(1.0, vec3(1, 1, 1), vec3(0, 0, 0));
    EXPECT_NEAR(cb.value(0, 0, vec3(0.2, 0.2, 0.2)).x(), 1); // even cell
    EXPECT_NEAR(cb.value(0, 0, vec3(1.2, 0.2, 0.2)).x(), 0); // odd cell
    // Lambertian samples texture with hit UVs.
    rng_seed(40);
    lambertian lamb(std::make_shared<checker>(1.0, vec3(1, 0, 0), vec3(0, 0, 1)));
    hit_record rec;
    rec.point = vec3(0.2, 0, 0);
    rec.normal = vec3(0, 1, 0);
    rec.front_face = true;
    vec3 att;
    ray sc;
    EXPECT_TRUE(lamb.scatter(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec, att, sc));
    EXPECT_NEAR(att.x(), 1); // even cell -> red
    rec.point = vec3(1.2, 0, 0);
    EXPECT_TRUE(lamb.scatter(ray(vec3(1.2, 1, 0), vec3(0, -1, 0)), rec, att, sc));
    EXPECT_NEAR(att.z(), 1); // odd cell -> blue
}

static void t_mipmaps() {
    test_current = "mipmaps";
    // 4x2 ramp: chain (4,2) -> (2,1) -> (1,1), L1 = box averages.
    std::vector<vec3> px;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 4; ++x)
            px.push_back(vec3(x / 3.0, y, 0));
    image_texture itex(4, 2, px);
    const auto &chain = itex.mip_chain();
    EXPECT_TRUE(chain.size() == 3);
    EXPECT_TRUE(chain[0].w == 4 && chain[0].h == 2);
    EXPECT_TRUE(chain[1].w == 2 && chain[1].h == 1);
    EXPECT_TRUE(chain[2].w == 1 && chain[2].h == 1);
    // L1 texel 0 = avg of x=0,1 rows 0,1: ((0+1/3)/2, 0.5, 0).
    EXPECT_NEAR(chain[1].px[0].x(), (0.0 + 1.0 / 3.0) / 2.0);
    EXPECT_NEAR(chain[1].px[0].y(), 0.5);
    // lod 0 sample is the exact legacy bilinear path.
    vec3 v0 = itex.value(0.3, 0.7, vec3());
    vec3 s0 = itex.sample(0.3, 0.7, vec3(), 0.01);
    EXPECT_TRUE(s0.x() == v0.x() && s0.y() == v0.y() && s0.z() == v0.z());
    // LOD selector: close stays 0, grows with distance.
    EXPECT_TRUE(mip_select(0.01, 128, 64, 225, 8.0) == 0);
    double l1 = mip_select(5.0, 128, 64, 225, 8.0);
    double l2 = mip_select(50.0, 128, 64, 225, 8.0);
    EXPECT_TRUE(l1 == 0 && l2 > 1.0 && l2 < 3.0);
    // Span calibration: huge-span ground never leaves L0, cube face climbs.
    EXPECT_TRUE(mip_select(200.0, 128, 64, 225, 628.0) == 0);
    EXPECT_TRUE(mip_select(10.0, 64, 64, 225, 0.7) > 1.0);
    // Shimmer kill: 4x4 checkerboard collapses to mean gray at distance.
    std::vector<vec3> cb;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            cb.push_back(((x + y) % 2 == 0) ? vec3(1, 1, 1) : vec3(0, 0, 0));
    image_texture checker(4, 4, cb);
    double lo = 1, hi = 0;
    for (double u : {0.1, 0.4, 0.7}) {
        // lod clamps to the 1x1 mean level (4px image needs far t).
        vec3 far = checker.sample(u, 0.3, vec3(), 5000.0);
        EXPECT_TRUE(fabs(far.x() - 0.5) < 1e-9); // exact global mean
        double n = checker.value(u, 0.3, vec3()).x();
        if (n < lo)
            lo = n;
        if (n > hi)
            hi = n;
    }
    EXPECT_TRUE(hi - lo > 0.4); // L0 still aliases across the same uvs
}

static void t_ggx() {
    test_current = "ggx";
    const double pi = 3.1415926535897932385;
    // NDF at normal incidence: 1/(PI a^2).
    EXPECT_NEAR(ggx::D(0.5, 1.0), 1.0 / (pi * 0.25));
    EXPECT_NEAR(ggx::lambda(0.5, 1.0), 0.0); // no shadowing overhead
    EXPECT_NEAR(ggx::lambda(0.0, 0.5), 0.0); // delta: G = 1
    EXPECT_NEAR(ggx::alpha_of(0.5), 0.25); // perceptual square
    // Roughness 0: H collapses to +z, L is the exact mirror.
    vec3 V(0.3, -0.4, 0.8660254);
    V = unit_vector(V);
    vec3 H;
    vec3 L = ggx::vndf_sample(0.0, V, 0.13, 0.71, H);
    EXPECT_NEAR(H.x(), 0); EXPECT_NEAR(H.y(), 0); EXPECT_NEAR(H.z(), 1);
    vec3 R = vec3(0, 0, 1) * (2.0 * V.z()) - V; // reflect incident about +z
    EXPECT_TRUE((L - R).length() < 1e-9);
    // White furnace: F0=1, r=0.5, V=+z. Absorbed draws count 0.
    rng_seed(99);
    double sum = 0;
    const int N = 20000;
    vec3 Vz(0, 0, 1);
    for (int i = 0; i < N; ++i) {
        vec3 Li = ggx::vndf_sample(0.25, Vz, random_double(), random_double(), H);
        if (Li.z() <= 0)
            continue;
        double F = 1.0; // F0=1 -> Schlick is 1
        sum += F * ggx::weight_ratio(0.25, 1.0, Li.z());
    }
    double alb = sum / N;
    EXPECT_TRUE(alb > 0.8 && alb <= 1.0); // energy conserved, none created
    // Metal scatter at roughness 0: exact mirror, albedo attenuation.
    metal chrome(vec3(0.8, 0.8, 0.8), 0.0);
    hit_record rec;
    rec.point = vec3(0, 0, 0);
    rec.normal = vec3(0, 0, 1);
    vec3 att;
    ray sc;
    rng_seed(7);
    EXPECT_TRUE(chrome.scatter(ray(vec3(0, 0, 1), vec3(0.2, 0, -1)), rec, att, sc));
    vec3 dir = unit_vector(vec3(0.2, 0, -1));
    vec3 refl = dir - vec3(0, 0, 1) * (2.0 * dir.z());
    EXPECT_TRUE((unit_vector(sc.direction()) - refl).length() < 1e-9);
    EXPECT_NEAR(att.x(), 0.8); // F0 at near-normal incidence
}

static void t_emissive() {
    test_current = "emissive";
    rng_seed(21);
    diffuse_light lamp(vec3(4, 4, 4));
    EXPECT_NEAR(lamp.emitted().x(), 4);
    hit_record rec;
    vec3 att; ray sc;
    EXPECT_TRUE(!lamp.scatter(ray(vec3(0,0,0), vec3(0,0,-1)), rec, att, sc));
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    quad q(vec3(-1, 0, -3), vec3(2, 0, 0), vec3(0, 2, 0), m);
    EXPECT_NEAR(q.area(), 4);
    vec3 p = q.sample_point();
    EXPECT_TRUE(p.x() >= -1 && p.x() <= 1 && p.y() >= 0 && p.y() <= 2);
    EXPECT_NEAR(p.z(), -3);
}

static void t_ppm() {
    test_current = "ppm";
    // 2x2 P3 round-trip through temp dir (no committed asset needed).
    std::string tmp = (std::filesystem::temp_directory_path() / "rt_tex_test.ppm").string();
    {
        std::ofstream f(tmp);
        f << "P3\n2 2\n255\n255 0 0 0 255 0 0 0 255 255 255 255\n";
    }
    ppm_io::image img;
    EXPECT_TRUE(ppm_io::read_ppm(tmp, img));
    EXPECT_TRUE(img.w == 2 && img.h == 2);
    EXPECT_NEAR(img.px[0].x(), 1); // top-left red
    EXPECT_NEAR(img.px[3].x(), 1); // bottom-right white
    EXPECT_NEAR(img.px[3].y(), 1);
    image_texture itex(img.w, img.h, img.px);
    // Bilinear: corners exact, center blends to gray.
    EXPECT_NEAR(itex.value(0.0, 1.0, vec3()).x(), 1); // top-left red
    vec3 mid = itex.value(0.5, 0.5, vec3());
    EXPECT_TRUE(fabs(mid.x() - 0.5) < 1e-9 && fabs(mid.y() - 0.5) < 1e-9 &&
                fabs(mid.z() - 0.5) < 1e-9);
    EXPECT_TRUE(!ppm_io::read_ppm(tmp + ".missing", img));
}

static void t_film() {
    test_current = "film";
    vec3 black = aces_approx(vec3(0, 0, 0));
    EXPECT_NEAR(black.x(), 0);
    double a05 = aces_approx(vec3(0.5, 0.5, 0.5)).x();
    double a1 = aces_approx(vec3(1, 1, 1)).x();
    double a4 = aces_approx(vec3(4, 4, 4)).x();
    EXPECT_TRUE(a05 < a1 && a1 < a4); // monotonic
    EXPECT_TRUE(a4 > 0.8 && a4 <= 1.0); // rolls off, never clips past 1
    EXPECT_NEAR(srgb_encode(0.0), 0.0);
    EXPECT_NEAR(srgb_encode(1.0), 1.0);
    // Exposure scales linear input: ev2 at half light == ev1 at full.
    vec3 t1 = tonemap(vec3(0.18, 0.18, 0.18), 2.0);
    vec3 t2 = tonemap(vec3(0.36, 0.36, 0.36), 1.0);
    EXPECT_NEAR(t1.x(), t2.x());
    // Light quad hue kept: bright white stays neutral, not clipped flat.
    vec3 lamp = tonemap(vec3(4, 4, 4), 1.0);
    EXPECT_TRUE(lamp.x() > 0.95 && lamp.y() > 0.95 && lamp.z() > 0.95);
}

static void t_denoise() {
    test_current = "denoise";
    // Flat field passes through untouched.
    std::vector<vec3> flat(64, vec3(0.5, 0.5, 0.5));
    auto f2 = bilateral_denoise(flat, 8, 8);
    EXPECT_NEAR(f2[27].x(), 0.5);
    EXPECT_NEAR(f2[27].y(), 0.5);
    // Step edge: contrast preserved (no cross-edge bleed past midpoint).
    std::vector<vec3> step(64);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x)
            step[(size_t)y * 8 + x] = (x < 4) ? vec3(0.2, 0.2, 0.2) : vec3(0.8, 0.8, 0.8);
    auto s2 = bilateral_denoise(step, 8, 8);
    EXPECT_TRUE(s2[(size_t)4 * 8 + 3].x() < 0.5);  // dark side stays dark
    EXPECT_TRUE(s2[(size_t)4 * 8 + 4].x() > 0.5);  // bright side stays bright
    EXPECT_TRUE(s2[(size_t)4 * 8 + 3].x() < s2[(size_t)4 * 8 + 4].x());
    // Noisy ramp: variance drops.
    rng_seed(60);
    std::vector<vec3> ramp(64);
    for (int i = 0; i < 64; ++i) {
        double g = (i % 8) / 7.0;
        double n = (random_double() - 0.5) * 0.2;
        ramp[i] = vec3(g + n, g + n, g + n);
    }
    // Detrend: residual energy around the true ramp must fall.
    auto r2 = bilateral_denoise(ramp, 8, 8);
    double raw_res = 0, den_res = 0;
    for (int i = 0; i < 64; ++i) {
        double g = (i % 8) / 7.0;
        raw_res += (ramp[i].x() - g) * (ramp[i].x() - g);
        den_res += (r2[i].x() - g) * (r2[i].x() - g);
    }
    EXPECT_TRUE(den_res < raw_res); // noise energy falls
}

static void t_aov() {
    test_current = "aov";
    rng_seed(70);
    auto m = std::make_shared<lambertian>(
        std::make_shared<checker>(1.0, vec3(1, 0, 0), vec3(0, 0, 1)));
    hittable_list w;
    w.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, m));
    vec3 alb, nrm;
    bool hit = false;
    // Ray at sphere front (0,0,-0.5): floor(-0.5)=-1 -> odd cell -> blue.
    first_hit_aov(ray(vec3(0, 0, 0), vec3(0, 0, -1)), w, alb, nrm, hit);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(alb.z(), 1);
    EXPECT_NEAR(nrm.z(), 1); // outward front normal
    // Miss: flag false, buffers stay zero.
    first_hit_aov(ray(vec3(0, 0, 0), vec3(0, 1, 0)), w, alb, nrm, hit);
    EXPECT_TRUE(!hit);
}

static void t_joint() {
    test_current = "joint";
    // Textured step: albedo edge red|blue, beauty = albedo*0.6 + noise.
    // Plain bilateral blurs the edge; joint must keep it and win residual.
    const int S = 16;
    rng_seed(71);
    std::vector<vec3> alb(S * S), nrm(S * S), truth(S * S), noisy(S * S);
    for (int y = 0; y < S; ++y)
        for (int x = 0; x < S; ++x) {
            vec3 a = (x < S / 2) ? vec3(1, 0, 0) : vec3(0, 0, 1);
            alb[(size_t)y * S + x] = a;
            nrm[(size_t)y * S + x] = vec3(0, 1, 0);
            truth[(size_t)y * S + x] = a * 0.6;
            double n = (random_double() - 0.5) * 0.3;
            noisy[(size_t)y * S + x] = a * 0.6 + vec3(n, n, n);
        }
    auto plain = bilateral_denoise(noisy, S, S);
    auto joint = joint_bilateral_denoise(noisy, alb, nrm, S, S);
    auto res = [&](const std::vector<vec3> &v) {
        double s = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            vec3 d = v[i] - truth[i];
            s += d.length_squared();
        }
        return s / v.size();
    };
    EXPECT_TRUE(res(joint) < res(plain)); // guides beat blind
    // Edge pixels stay on their own side (no cross-bleed past middle).
    EXPECT_TRUE(joint[(size_t)8 * S + 7].x() > joint[(size_t)8 * S + 7].z());
    EXPECT_TRUE(joint[(size_t)8 * S + 8].z() > joint[(size_t)8 * S + 8].x());
}

static void t_stb() {
    test_current = "stb";
    // Exact sRGB inverse (not gamma 2.2): 0.5 -> ~0.214.
    EXPECT_TRUE(fabs(stb_loader::srgb_to_linear(0.5) - 0.2140) < 0.002);
    EXPECT_NEAR(stb_loader::srgb_to_linear(0.0), 0.0);
    EXPECT_NEAR(stb_loader::srgb_to_linear(1.0), 1.0);
    // Hand-written 2x2 BMP (54B header, BGR, bottom-first, padded rows).
    std::string bmp = (std::filesystem::temp_directory_path() / "rt_2x2.bmp").string();
    {
        unsigned char hdr[54] = {};
        hdr[0] = 'B';
        hdr[1] = 'M';
        hdr[2] = 70;
        hdr[10] = 54;
        hdr[14] = 40;
        hdr[18] = 2;
        hdr[22] = 2;
        hdr[26] = 1;
        hdr[28] = 24;
        hdr[34] = 16;
        // Bottom row: red, green. Top row: blue, white. (BGR triples.)
        unsigned char px[16] = {0, 0, 255, 0, 255, 0, 0, 0,
                                255, 0, 0, 255, 255, 255, 0, 0};
        std::ofstream f(bmp, std::ios::binary);
        f.write((char *)hdr, 54);
        f.write((char *)px, 16);
    }
    ppm_io::image img;
    EXPECT_TRUE(stb_loader::load_image(bmp, img));
    EXPECT_TRUE(img.w == 2 && img.h == 2);
    EXPECT_NEAR(img.px[0].x(), 0); // top-left blue (linearized, still ~0 red)
    EXPECT_TRUE(img.px[0].z() > 0.9);
    EXPECT_TRUE(img.px[1].x() > 0.9); // top-right white
    EXPECT_TRUE(img.px[3].y() > 0.9); // bottom-right green
    // Committed JPEG asset: gradient red→x, green→y, flat blue.
    ppm_io::image photo;
    EXPECT_TRUE(stb_loader::load_image("assets/photo_test.jpg", photo));
    EXPECT_TRUE(photo.w == 128 && photo.h == 64);
    EXPECT_TRUE(photo.px[0].x() < photo.px[127].x()); // red rises with x
    EXPECT_TRUE(photo.px[0].y() < photo.px[(size_t)63 * 128].y()); // green rises
    EXPECT_TRUE(!stb_loader::load_image(bmp + ".missing", photo));
}

static void t_pfm() {
    test_current = "pfm";
    // 2x1 film: bottom-left HDR (4, 0.5, 0.125), top-right LDR gray.
    std::vector<vec3> fb = {vec3(4, 0.5, 0.125), vec3(0, 0, 0), vec3(0.5, 0.5, 0.5),
                            vec3(0, 0, 0)};
    std::string pfm = (std::filesystem::temp_directory_path() / "rt_hdr.pfm").string();
    EXPECT_TRUE(write_pfm(pfm.c_str(), fb, 2, 2));
    EXPECT_TRUE(!write_pfm(pfm.c_str(), fb, 3, 2)); // size mismatch refuses
    std::ifstream f(pfm, std::ios::binary);
    std::string magic;
    int w = 0, h = 0;
    double scale = 0;
    f >> magic >> w >> h >> scale;
    EXPECT_TRUE(magic == "PF" && w == 2 && h == 2 && scale < 0); // little-endian
    f.get(); // single newline after scale
    float rgb[3] = {};
    f.read((char *)rgb, sizeof rgb); // first triple = bottom-left, raw linear
    EXPECT_TRUE(fabs(rgb[0] - 4.0f) < 1e-6 && fabs(rgb[1] - 0.5f) < 1e-6 &&
                fabs(rgb[2] - 0.125f) < 1e-6);
}

static void t_compare() {
    test_current = "compare";
    // Identical -> zeros.
    std::vector<uint8_t> a = {10, 20, 30, 40, 50, 60};
    diff_stats s0 = compare_images(a, a, 2, 1);
    EXPECT_NEAR(s0.mean_abs, 0);
    EXPECT_NEAR(s0.max_abs, 0);
    EXPECT_NEAR(s0.frac_over, 0);
    // One byte +9 over 6: mean 1.5, max 9, over8 = 1/6.
    std::vector<uint8_t> b = {10, 20, 30, 40, 50, 69};
    diff_stats s1 = compare_images(a, b, 2, 1);
    EXPECT_NEAR(s1.mean_abs, 1.5);
    EXPECT_NEAR(s1.max_abs, 9);
    EXPECT_TRUE(fabs(s1.frac_over - 1.0 / 6) < 1e-9);
    // Heatmap: per-pixel mean diff x gain (9/3*4=12 on changed pixel).
    auto heat = diff_heatmap(a, b, 2, 1);
    EXPECT_TRUE(heat[0] == 0 && heat[1] == 0 && heat[2] == 0);
    EXPECT_TRUE(heat[3] == 12 && heat[4] == 12 && heat[5] == 12);
    // Size mismatch: safe zeros, no crash.
    diff_stats s2 = compare_images(a, {1, 2}, 2, 1);
    EXPECT_NEAR(s2.mean_abs, 0);
}

void run_shading_tests() {
    t_materials();
    t_texture();
    t_mipmaps();
    t_ggx();
    t_emissive();
    t_ppm();
    t_pfm();
    t_film();
    t_denoise();
    t_aov();
    t_joint();
    t_stb();
    t_compare();
}
