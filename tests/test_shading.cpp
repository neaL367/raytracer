// Shading + output tests: materials, textures, emissives, PPM, film.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/texture.h"
#include "geometry/hittable.h"
#include "geometry/quad.h"
#include "material/material.h"
#include "io/ppm_image.h"
#include "io/denoise.h"
#include "output/film.h"

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
    EXPECT_NEAR(itex.value(0.25, 0.75, vec3()).x(), 1); // uv->top-left red
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

void run_shading_tests() {
    t_materials();
    t_texture();
    t_emissive();
    t_ppm();
    t_film();
    t_denoise();
}
