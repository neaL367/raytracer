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
#include "scene/scene.h"
#include "integrator/integrator.h"
#include "integrator/pdf.h"
#include "core/env.h"
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

static void t_glass_rough() {
    test_current = "glass_rough";
    hit_record rec;
    rec.point = vec3(0, 0, 0);
    rec.normal = vec3(0, 1, 0);
    rec.front_face = true;
    vec3 att, att0;
    ray sc, sc0;
    // Roughness 0 == legacy delta path bit-exact (same seed, same draws).
    dielectric smooth(1.5), rough0(1.5, 0.0);
    rng_seed(55);
    EXPECT_TRUE(rough0.scatter(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec, att, sc));
    rng_seed(55);
    EXPECT_TRUE(smooth.scatter(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec, att0, sc0));
    EXPECT_TRUE((sc.direction() - sc0.direction()).length() == 0);
    EXPECT_TRUE((att - att0).length() == 0);
    // Rough glass: weights bounded by 1 (G2/G1), dirs valid + unit.
    dielectric rg(1.5, 0.4);
    rng_seed(56);
    double wmax = 0;
    int nref = 0, ntr = 0;
    for (int i = 0; i < 5000; ++i) {
        vec3 a;
        ray s;
        if (!rg.scatter(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec, a, s))
            continue; // degenerate lobe absorbed
        EXPECT_TRUE(a.x() <= 1.0 && a.x() >= 0); // energy-safe per sample
        if (a.x() > wmax)
            wmax = a.x();
        EXPECT_TRUE(fabs(s.direction().length() - 1.0) < 1e-9);
        if (s.direction().y() > 0)
            ++nref; // reflected lobe above
        else
            ++ntr; // transmitted lobe below
    }
    EXPECT_TRUE(wmax <= 1.0 && wmax > 0.5); // nontrivial, never created
    EXPECT_TRUE(nref > 100 && ntr > 100); // both lobes sampled
    // TIR from inside at grazing, near-zero roughness: H ~= N forces
    // sinT2 > 1 for every draw -> pure reflection, stays inside.
    dielectric polished(1.5, 0.01);
    rec.normal = vec3(0, -1, 0); // backface: against-ray, into glass
    rec.front_face = false;
    rng_seed(57);
    for (int i = 0; i < 20; ++i) {
        vec3 at2;
        ray s2;
        EXPECT_TRUE(polished.scatter(
            ray(vec3(0, 0, 0), unit_vector(vec3(1, 0.05, 0))), rec, at2, s2));
        EXPECT_TRUE(s2.direction().y() < 0); // reflected, never transmits
    }
    rec.normal = vec3(0, 1, 0);
    rec.front_face = true;
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

static void t_aniso() {
    test_current = "aniso";
    const double pi = 3.1415926535897932385;
    // NDF at normal incidence: 1/(PI ax ay).
    EXPECT_NEAR(ggx::D_aniso(0.25, 0.5, vec3(0, 0, 1)), 1.0 / (pi * 0.25 * 0.5));
    EXPECT_NEAR(ggx::lambda_aniso(0.25, 0.5, vec3(0, 0, 1)), 0.0);
    // ax == ay reduces bit-exactly to the isotropic formulas.
    vec3 V = unit_vector(vec3(0.3, -0.4, 0.8660254));
    vec3 H;
    vec3 La = ggx::vndf_aniso(0.25, 0.25, V, 0.13, 0.71, H);
    vec3 Ha;
    vec3 Li2 = ggx::vndf_sample(0.25, V, 0.13, 0.71, Ha);
    EXPECT_TRUE((La - Li2).length() == 0 && (H - Ha).length() == 0);
    EXPECT_NEAR(ggx::D_aniso(0.25, 0.25, H), ggx::D(0.25, H.z()));
    EXPECT_NEAR(ggx::weight_ratio_aniso(0.25, 0.25, V, Li2),
                ggx::weight_ratio(0.25, V.z(), Li2.z()));
    // White furnace, stretched lobe: energy conserved, none created.
    rng_seed(98);
    double sum = 0;
    const int N = 20000;
    vec3 Vz(0, 0, 1);
    for (int i = 0; i < N; ++i) {
        vec3 L = ggx::vndf_aniso(0.09, 0.36, Vz, random_double(), random_double(), H);
        if (L.z() <= 0)
            continue;
        sum += ggx::weight_ratio_aniso(0.09, 0.36, Vz, L); // F0=1 -> F=1
    }
    double alb = sum / N;
    EXPECT_TRUE(alb > 0.8 && alb <= 1.0);
    // Aniso metal scatters with tangent frame; type-10 export.
    metal brushed(vec3(0.8, 0.8, 0.8), 0.15, 0.5);
    hit_record rec;
    rec.point = vec3(0, 0, 0);
    rec.normal = vec3(0, 0, 1);
    rec.tangent = vec3(1, 0, 0);
    rec.has_tangent = true;
    vec3 att;
    ray sc;
    rng_seed(8);
    EXPECT_TRUE(brushed.scatter(ray(vec3(0, 0, 1), vec3(0.2, 0, -1)), rec, att, sc));
    EXPECT_TRUE(att.x() <= 1.0 && att.x() > 0); // bounded weight
    EXPECT_TRUE(fabs(sc.direction().length() - 1.0) < 1e-6); // unit out
    float alb4[4] = {}, alb24[4] = {}, emit4[4] = {}, prm4[4] = {};
    EXPECT_TRUE(brushed.export_gpu(alb4, alb24, emit4, prm4));
    EXPECT_TRUE(prm4[0] == 10.0f);
    EXPECT_TRUE(fabs(prm4[1] - 0.15f) < 1e-6 && fabs(prm4[2] - 0.5f) < 1e-6);
    // Missing tangent falls back without crashing (unit X lobe).
    rec.has_tangent = false;
    rng_seed(9);
    EXPECT_TRUE(brushed.scatter(ray(vec3(0, 0, 1), vec3(0, 0, -1)), rec, att, sc));
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
    // Depth probe: front hit at t=0.5, miss leaves the out-param alone.
    double depth = -7;
    first_hit_aov(ray(vec3(0, 0, 0), vec3(0, 0, -1)), w, alb, nrm, hit, &depth);
    EXPECT_TRUE(hit);
    EXPECT_NEAR(depth, 0.5);
    double depth2 = -7;
    first_hit_aov(ray(vec3(0, 0, 0), vec3(0, 1, 0)), w, alb, nrm, hit, &depth2);
    EXPECT_TRUE(!hit);
    EXPECT_NEAR(depth2, -7);
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

static void t_noise() {
    test_current = "noise";
    // Lattice hash is deterministic and bounded.
    EXPECT_TRUE(value_noise::lattice_hash(3, 1, 4) == value_noise::lattice_hash(3, 1, 4));
    EXPECT_TRUE(value_noise::lattice_hash(3, 1, 4) != value_noise::lattice_hash(4, 1, 4));
    double u0 = value_noise::lattice_unit(0, 0, 0);
    EXPECT_TRUE(u0 >= 0.0 && u0 <= 1.0);
    EXPECT_NEAR(u0, 0.0); // null lattice hashes to zero by construction
    // Negative lattice coords stay in range (no UB, no table).
    double un = value_noise::lattice_unit(-3, -1, -4);
    EXPECT_TRUE(un >= 0.0 && un <= 1.0);
    // Golden probe: smoothed noise at a fixed point (implementation golden).
    double g = value_noise::at(vec3(3.14, 4.2, 7.0));
    EXPECT_TRUE(g >= 0.0 && g <= 1.0);
    EXPECT_NEAR(g, 0.339232831391592);
    EXPECT_NEAR(value_noise::at(vec3(3.14, 4.2, 7.0)), g); // replay bit-exact
    // Depth-1 turbulence is |noise| (weight normalization exact).
    double a = value_noise::at(vec3(1.7, 2.3, 0.4));
    EXPECT_NEAR(value_noise::turb(vec3(1.7, 2.3, 0.4), 1), std::fabs(a));
    EXPECT_TRUE(value_noise::turb(vec3(1.7, 2.3, 0.4), 7) >= 0.0);
    EXPECT_TRUE(value_noise::turb(vec3(1.7, 2.3, 0.4), 7) <= 1.0);
    // Modes: raw/turb/marble all lerp between the two colors.
    noise_texture raw(2.0, 4, 0, vec3(0, 0, 0), vec3(1, 1, 1));
    noise_texture tb(2.0, 4, 1, vec3(0, 0, 0), vec3(1, 1, 1));
    noise_texture mb(2.0, 4, 2, vec3(0, 0, 0), vec3(1, 1, 1));
    vec3 p(0.5, 1.5, 2.5);
    for (auto *t : {&raw, &tb, &mb}) {
        vec3 v = t->value(0, 0, p);
        EXPECT_TRUE(v.x() >= 0.0 && v.x() <= 1.0);
        EXPECT_TRUE(fabs(v.x() - v.y()) < 1e-12 && fabs(v.y() - v.z()) < 1e-12);
    }
    // Marble self-consistency: sine band over depth-7 turbulence.
    double t7 = value_noise::turb(p * 2.0, 4);
    double f = 0.5 * (1.0 + std::sin(2.0 * p.z() + 10.0 * t7));
    EXPECT_NEAR(mb.value(0, 0, p).x(), f);
    // GPU export: type 9 with (freq, depth, mode).
    auto ntex = std::make_shared<noise_texture>(4.0, 7, 2, vec3(0.85, 0.87, 0.9),
                                                vec3(0.05, 0.15, 0.45));
    lambertian nl(ntex);
    float alb[4] = {}, alb2[4] = {}, emit[4] = {}, prm[4] = {};
    EXPECT_TRUE(nl.export_gpu(alb, alb2, emit, prm));
    EXPECT_TRUE(prm[0] == 9 && prm[1] == 4.0f && prm[2] == 7.0f && prm[3] == 2.0f);
    EXPECT_TRUE(fabs(alb[0] - 0.85) < 1e-6); // float export vs double literal
    EXPECT_TRUE(fabs(alb2[2] - 0.45) < 1e-6);
    // Scatter path tints by the noise value (not solid passthrough).
    hit_record rec;
    rec.point = p;
    rec.normal = vec3(0, 1, 0);
    rec.front_face = true;
    vec3 att;
    ray sc;
    rng_seed(41);
    EXPECT_TRUE(nl.scatter(ray(vec3(0.5, 2.5, 2.5), vec3(0, -1, 0)), rec, att, sc));
    EXPECT_NEAR(att.x(), ntex->value(0, 0, p).x());
}

static void t_pdf() {
    test_current = "pdf";
    auto lamp = std::make_shared<diffuse_light>(vec3(3, 3, 3));
    auto quad_lamp =
        std::make_shared<quad>(vec3(-1, 2, -1), vec3(2, 0, 0), vec3(0, 0, 2), lamp);
    hittable_list world;
    world.add(quad_lamp);
    std::vector<light> lights{light(quad_lamp)};
    vec3 origin(0, 0, 0), n(0, 1, 0);
    // Straight up strikes the 2x2 lamp: dist=2, A=4, cos=1 -> pdf 1.
    EXPECT_NEAR(direction_pdf::nee_value(world, lights, origin, vec3(0, 1, 0), 0.0), 1.0);
    // Sideways misses everything -> 0. Downward misses -> 0.
    EXPECT_NEAR(direction_pdf::nee_value(world, lights, origin, vec3(1, 0, 0), 0.0), 0.0);
    EXPECT_NEAR(direction_pdf::nee_value(world, lights, origin, vec3(0, -1, 0), 0.0),
                0.0);
    // Matte (non-emissive) strike reports 0 even dead-on.
    hittable_list matte_world;
    matte_world.add(std::make_shared<quad>(vec3(-1, 2, -1), vec3(2, 0, 0), vec3(0, 0, 2),
                                           std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5))));
    EXPECT_NEAR(direction_pdf::nee_value(matte_world, lights, origin, vec3(0, 1, 0), 0.0),
                0.0);
    // Cosine lobe: normal incidence 1/PI, grazing 0.
    const double pi = 3.1415926535897932385;
    EXPECT_NEAR(direction_pdf::cosine_value(vec3(0, 1, 0), n), 1.0 / pi);
    EXPECT_NEAR(direction_pdf::cosine_value(vec3(1, 0, 0), n), 0.0);
    // Mixture is the exact 50/50 blend.
    double c = direction_pdf::cosine_value(vec3(0, 1, 0), n);
    double l = direction_pdf::nee_value(world, lights, origin, vec3(0, 1, 0), 0.0);
    EXPECT_NEAR(direction_pdf::mixture_value(world, lights, origin, vec3(0, 1, 0), n, 0.0),
                0.5 * c + 0.5 * l);
    // Sampler: unit dirs, self-consistent pdf, deterministic replay.
    rng_seed(101);
    double p1 = -1;
    vec3 d1 = direction_pdf::sample_mixture(world, lights, origin, n, 0.0, p1);
    EXPECT_TRUE(fabs(d1.length() - 1.0) < 1e-9 && p1 > 0);
    EXPECT_NEAR(p1, direction_pdf::mixture_value(world, lights, origin, d1, n, 0.0));
    rng_seed(101);
    double p2 = -1;
    vec3 d2 = direction_pdf::sample_mixture(world, lights, origin, n, 0.0, p2);
    EXPECT_NEAR((d1 - d2).length(), 0.0);
    EXPECT_NEAR(p1, p2);
    // Empty lights: pure cosine branch, still positive density upward.
    std::vector<light> none;
    rng_seed(102);
    double p3 = -1;
    vec3 d3 = direction_pdf::sample_mixture(world, none, origin, n, 0.0, p3);
    EXPECT_TRUE(p3 > 0 && dot(d3, n) > 0);
    // Power heuristic: equal densities split half; dominant takes ~all.
    EXPECT_NEAR(direction_pdf::power_weight(1.0, 1.0), 0.5);
    EXPECT_NEAR(direction_pdf::power_weight(3.0, 1.0), 0.9);
    EXPECT_NEAR(direction_pdf::power_weight(0.0, 0.0), 0.0);
    // Traceless reverse matches the traced reverse on the same strike.
    double traced = direction_pdf::nee_value(world, lights, origin, vec3(0, 1, 0), 0.0);
    double direct = direction_pdf::nee_value_for_hit(lights, lamp, vec3(0, 2, 0), origin,
                                                     0.0);
    EXPECT_NEAR(traced, direct);
    EXPECT_NEAR(direct, 1.0);
    // Direction densities: lambertian cosine, isotropic uniform, delta zero.
    hit_record mrec;
    mrec.normal = n;
    lambertian lamb0(vec3(0.5, 0.5, 0.5));
    EXPECT_NEAR(lamb0.direction_pdf(vec3(0, 1, 0), mrec), 1.0 / pi);
    isotropic iso(vec3(0.9, 0.9, 0.9));
    EXPECT_NEAR(iso.direction_pdf(vec3(1, 0, 0), mrec), 1.0 / (4.0 * pi));
    metal mirror(vec3(0.8, 0.8, 0.8), 0.0);
    EXPECT_NEAR(mirror.direction_pdf(vec3(0, 1, 0), mrec), 0.0);
}

static void t_env() {
    test_current = "env";
    // Sky gradient endpoints: white below, blue above (legacy look).
    EXPECT_NEAR(env_light::sky(vec3(0, -1, 0)).x(), 1.0);
    EXPECT_NEAR(env_light::sky(vec3(0, 1, 0)).z(), 1.0);
    // Sun disc: center ~30x white, rim falls to sky.
    vec3 sun = env_light::sun_dir();
    EXPECT_TRUE(fabs(sun.length() - 1.0) < 1e-12);
    vec3 center = env_light::radiance(sun);
    EXPECT_TRUE(center.x() > 25.0 && center.x() < 35.0);
    vec3 anti = env_light::radiance(-sun);
    vec3 anti_sky = env_light::sky(-sun);
    EXPECT_NEAR(anti.x(), anti_sky.x()); // no sun behind: pure sky
    // Uniform-sphere pdf is 1/4PI; sampler returns unit dirs, replayable.
    const double pi = 3.1415926535897932385;
    EXPECT_NEAR(env_light::sample_pdf(), 1.0 / (4.0 * pi));
    vec3 d = env_light::sample_dir(0.0, 0.0); // north pole, exact
    EXPECT_NEAR(d.z(), 1.0);
    vec3 d2 = env_light::sample_dir(0.25, 0.5);
    EXPECT_TRUE(fabs(d2.length() - 1.0) < 1e-9);
    // Env-off miss is the legacy sky bit-exact (frozen path).
    integrator tracer;
    hittable_list empty;
    std::vector<light> none;
    std::vector<std::shared_ptr<hittable>> nomedia;
    rng_seed(200);
    vec3 miss = tracer.Li(ray(vec3(0, 0, 0), vec3(0, 0.5, -1)), empty, none, 50,
                           nomedia, false);
    vec3 expect = env_light::sky(vec3(0, 0.5, -1));
    EXPECT_NEAR(miss.x(), expect.x());
    EXPECT_NEAR(miss.y(), expect.y());
    EXPECT_NEAR(miss.z(), expect.z());
    // Env-on primary miss sees the sun when aimed at it.
    rng_seed(201);
    vec3 sunshot =
        tracer.Li(ray(vec3(0, 0, 0), sun), empty, none, 50, nomedia, true);
    EXPECT_TRUE(sunshot.x() > 20.0);
    rng_seed(202);
    vec3 antishot =
        tracer.Li(ray(vec3(0, 0, 0), -sun), empty, none, 50, nomedia, true);
    EXPECT_NEAR(antishot.x(), env_light::sky(-sun).x());
    // Scene flag defaults off (frozen), opts in.
    scene_data s0 = build_default(16.0 / 9.0, 0.0);
    EXPECT_TRUE(!s0.env_light);
    scene_data s1 = build_default(16.0 / 9.0, 0.0, 0, 0, 0, 0, false, true);
    EXPECT_TRUE(s1.env_light);
}

void run_shading_tests() {
    t_materials();
    t_texture();
    t_mipmaps();
    t_noise();
    t_pdf();
    t_env();
    t_ggx();
    t_aniso();
    t_glass_rough();
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
