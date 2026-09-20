// Dependency-free tests. Same 30-line registry, deterministic seeds.
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "core/onb.h"
#include "core/aabb.h"
#include "core/texture.h"
#include "core/obj_loader.h"
#include "io/ppm_image.h"
#include "output/film.h"
#include "accel/bvh.h"
#include "integrator/integrator.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/sphere.h"
#include "geometry/triangle.h"
#include "geometry/quad.h"
#include "material/material.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static int checks = 0, failures = 0;
static const char *cur = "";
static void check(bool c, const char *e, int l) {
    ++checks;
    if (!c) { ++failures; std::printf("  FAIL %s:%d: %s\n", cur, l, e); }
}
static bool near(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}
#define EXPECT_TRUE(x) check((x), #x, __LINE__)
#define EXPECT_NEAR(a, b) check(near((a), (b)), #a " ~= " #b, __LINE__)

static void t_vec3() {
    cur = "vec3";
    vec3 a(1, 2, 3), b(4, -1, 0.5);
    vec3 s = a + b;
    EXPECT_NEAR(s.x(), 5); EXPECT_NEAR(s.y(), 1); EXPECT_NEAR(s.z(), 3.5);
    EXPECT_NEAR(dot(a, b), 3.5);
    EXPECT_NEAR(cross(vec3(1,0,0), vec3(0,1,0)).z(), 1);
    EXPECT_NEAR(unit_vector(vec3(0,3,4)).length(), 1);
    EXPECT_NEAR(reflect(vec3(1,-1,0), vec3(0,1,0)).y(), 1); // mirror
}
static void t_ray() {
    cur = "ray";
    EXPECT_NEAR(ray(vec3(0,0,0), vec3(0,0,-1)).at(0.5).z(), -0.5);
}
static void t_camera() {
    cur = "camera";
    vec3 u = unit_vector(camera().get_ray(0.5, 0.5).direction());
    EXPECT_NEAR(u.x(), 0); EXPECT_NEAR(u.y(), 0); EXPECT_NEAR(u.z(), -1);
}
static void t_sphere() {
    cur = "sphere";
    rng_seed(1);
    auto m = std::make_shared<lambertian>(vec3(0.7, 0.3, 0.3));
    sphere s(vec3(0,0,-1), 0.5, m);
    hit_record rec;
    EXPECT_TRUE(s.hit(ray(vec3(0,0,0), vec3(0,0,-1)), 0.001, 1e30, rec));
    EXPECT_NEAR(rec.t, 0.5);
    EXPECT_NEAR(rec.normal.z(), 1);
    EXPECT_TRUE(rec.mat == m);
    EXPECT_TRUE(!s.hit(ray(vec3(0,0,0), vec3(0,1,0)), 0.001, 1e30, rec));
}
static void t_list() {
    cur = "list";
    rng_seed(2);
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hittable_list w;
    w.add(std::make_shared<sphere>(vec3(0,0,-2), 0.5, m)); // far
    w.add(std::make_shared<sphere>(vec3(0,0,-1), 0.5, m)); // near
    hit_record rec;
    EXPECT_TRUE(w.hit(ray(vec3(0,0,0), vec3(0,0,-1)), 0.001, 1e30, rec));
    EXPECT_NEAR(rec.t, 0.5); // closest wins
}
static void t_triangle_quad() {
    cur = "triquad";
    rng_seed(3);
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    triangle t(vec3(-1,-1,-2), vec3(1,-1,-2), vec3(0,1,-2), m);
    hit_record rec;
    EXPECT_TRUE(t.hit(ray(vec3(0,0,0), vec3(0,0,-1)), 0.001, 1e30, rec));
    EXPECT_NEAR(rec.t, 2.0);
    EXPECT_TRUE(!t.hit(ray(vec3(0,0,0), vec3(0,1,0)), 0.001, 1e30, rec));
    EXPECT_TRUE(!t.hit(ray(vec3(0,0,0), vec3(1,0,0)), 0.001, 1e30, rec)); // parallel-ish miss
    quad q(vec3(-1,-1,-3), vec3(2,0,0), vec3(0,2,0), m);
    EXPECT_TRUE(q.hit(ray(vec3(0,0,0), vec3(0,0,-1)), 0.001, 1e30, rec));
    EXPECT_NEAR(rec.normal.z(), 1);
    EXPECT_TRUE(!q.hit(ray(vec3(5,5,0), vec3(0,0,-1)), 0.001, 1e30, rec)); // outside
}
static void t_materials() {
    cur = "materials";
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

static void t_sampler() {
    cur = "sampler";
    rng_seed(10);
    auto j = jitter_offsets(50);
    EXPECT_TRUE((int)j.size() == 50);
    for (auto [ox, oy] : j)
        EXPECT_TRUE(ox >= 0 && ox < 1 && oy >= 0 && oy < 1);
    rng_seed(10);
    auto s = stratified_offsets(4); // 16 samples, 4x4 cells
    EXPECT_TRUE((int)s.size() == 16);
    int cells[4][4] = {};
    for (auto [ox, oy] : s) {
        EXPECT_TRUE(ox >= 0 && ox < 1 && oy >= 0 && oy < 1);
        int cx = (int)(ox * 4), cy = (int)(oy * 4);
        if (cx > 3) cx = 3;
        if (cy > 3) cy = 3;
        cells[cy][cx]++;
    }
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            EXPECT_TRUE(cells[y][x] == 1); // each cell exactly once
    auto one = pixel_samples(1);
    EXPECT_TRUE(one.size() == 1 && near(one[0].first, 0.5) && near(one[0].second, 0.5));
    EXPECT_TRUE((int)pixel_samples(7).size() == 7); // non-square -> jitter
}

// Monte Carlo: integrate f(x,y)=x+y over unit square (truth=1.0).
// 20 seeded trials each; stratified variance must beat jitter.
static double trial_estimate(bool stratified, unsigned seed) {
    rng_seed(seed);
    double sum = 0;
    if (stratified) {
        for (auto [ox, oy] : stratified_offsets(4))
            sum += ox + oy;
        return sum / 16;
    }
    for (auto [ox, oy] : jitter_offsets(16))
        sum += ox + oy;
    return sum / 16;
}
static void t_montecarlo() {
    cur = "montecarlo";
    double ms = 0, mj = 0;
    double vs = 0, vj = 0;
    const int T = 20;
    double es[T], ej[T];
    for (int t = 0; t < T; ++t) {
        es[t] = trial_estimate(true, 100 + (unsigned)t);
        ej[t] = trial_estimate(false, 200 + (unsigned)t);
        ms += es[t]; mj += ej[t];
    }
    ms /= T; mj /= T;
    for (int t = 0; t < T; ++t) {
        vs += (es[t] - ms) * (es[t] - ms);
        vj += (ej[t] - mj) * (ej[t] - mj);
    }
    vs /= T; vj /= T;
    EXPECT_TRUE(fabs(ms - 1.0) < 0.05); // converges to truth
    EXPECT_TRUE(fabs(mj - 1.0) < 0.15);
    EXPECT_TRUE(vs < vj); // strata win on variance
}

static void t_onb_cosine() {
    cur = "onb";
    onb frame;
    frame.build_from_w(vec3(0, 1, 0));
    EXPECT_NEAR(frame.u.length(), 1);
    EXPECT_NEAR(frame.v.length(), 1);
    EXPECT_NEAR(frame.w.length(), 1);
    EXPECT_NEAR(dot(frame.u, frame.v), 0);
    EXPECT_NEAR(dot(frame.v, frame.w), 0);
    EXPECT_NEAR(dot(frame.w, frame.u), 0);
    rng_seed(20);
    double mean_z = 0;
    const int N = 10000;
    for (int i = 0; i < N; ++i)
        mean_z += random_cosine_direction().z();
    mean_z /= N;
    EXPECT_TRUE(fabs(mean_z - 0.6667) < 0.02); // pdf z/PI integrates so
    EXPECT_NEAR(cosine_pdf(1.0), 1.0 / 3.1415926535897932385);
    EXPECT_NEAR(cosine_pdf(-0.5), 0.0);
}
static void t_emissive() {
    cur = "emissive";
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
static void t_defocus() {
    cur = "defocus";
    rng_seed(22);
    camera pin; // default pinhole
    camera pin2(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90.0, 16.0/9.0, 0.0, 1.0);
    rng_seed(22);
    vec3 a = pin2.get_ray(0.25, 0.75).direction();
    rng_seed(22);
    vec3 b = pin.get_ray(0.25, 0.75).direction();
    EXPECT_NEAR(a.x(), b.x()); // aperture 0 == pinhole bit-exact
    EXPECT_NEAR(a.y(), b.y());
    EXPECT_NEAR(a.z(), b.z());
    camera wide(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90.0, 16.0/9.0, 2.0, 1.0);
    rng_seed(23);
    vec3 o1 = wide.get_ray(0.5, 0.5).origin();
    vec3 o2 = wide.get_ray(0.5, 0.5).origin();
    EXPECT_TRUE(o1.length() <= 1.0 + 1e-9); // on lens disk radius 1
    EXPECT_TRUE(o2.length() <= 1.0 + 1e-9);
    EXPECT_TRUE((o1 - o2).length() > 1e-9); // origins actually spread
}
static void t_rr() {
    cur = "rr";
    rng_seed(24);
    // Mirror of integrator RR: q=0.5 fixed throughput, mean preserved.
    double sum = 0;
    const int T = 20000;
    for (int i = 0; i < T; ++i) {
        double tput = 0.5;
        double q = tput < 0.95 ? tput : 0.95;
        double v = (random_double() > q) ? 0.0 : tput / q;
        sum += v;
    }
    EXPECT_TRUE(fabs(sum / T - 0.5) < 0.02); // unbiased termination
}

static void t_aabb() {
    cur = "aabb";
    aabb b(vec3(-1, -1, -1), vec3(1, 1, 1));
    hit_record dummy;
    EXPECT_TRUE(b.hit(ray(vec3(0, 0, 5), vec3(0, 0, -1)), 0.001, 1e30));
    EXPECT_TRUE(!b.hit(ray(vec3(0, 0, 5), vec3(0, 1, 0)), 0.001, 1e30));
    // Parallel ray inside slab survives; outside rejected.
    EXPECT_TRUE(b.hit(ray(vec3(0, 0, 0), vec3(1, 0, 0)), 0.001, 1e30));
    EXPECT_TRUE(!b.hit(ray(vec3(0, 5, 0), vec3(1, 0, 0)), 0.001, 1e30));
    aabb c(vec3(0, 0, 0), vec3(2, 2, 2));
    aabb u = aabb::surrounding(b, c);
    EXPECT_NEAR(u.minimum.x(), -1);
    EXPECT_NEAR(u.maximum.x(), 2);
    (void)dummy;
}
static void t_bvh() {
    cur = "bvh";
    rng_seed(30);
    auto m = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));
    std::vector<std::shared_ptr<hittable>> objs;
    objs.push_back(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, m));
    objs.push_back(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, m));
    objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, m));
    objs.push_back(std::make_shared<triangle>(vec3(-1, -1, -2), vec3(1, -1, -2),
                                              vec3(0, 1, -2), m));
    objs.push_back(std::make_shared<quad>(vec3(-1, -1, -3), vec3(2, 0, 0),
                                          vec3(0, 2, 0), m));
    hittable_list list;
    for (auto &o : objs)
        list.add(o);
    bvh_node tree(objs, 0, objs.size());
    int agree = 0;
    for (int k = 0; k < 200; ++k) {
        vec3 o(random_double(-2, 2), random_double(-2, 2), random_double(-1, 1));
        ray r(o, random_unit_vector());
        hit_record rl, rb;
        bool hl = list.hit(r, 0.001, 1e30, rl);
        bool hb = tree.hit(r, 0.001, 1e30, rb);
        EXPECT_TRUE(hl == hb);
        if (hl && hb) {
            EXPECT_TRUE(fabs(rl.t - rb.t) < 1e-9);
            EXPECT_TRUE(rl.mat == rb.mat);
            ++agree;
        }
    }
    EXPECT_TRUE(agree > 50); // scene actually hit, not all misses
}

static void t_texture_uv() {
    cur = "texture";
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

    cur = "uv";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    sphere s(vec3(0, 0, 0), 1.0, m);
    hit_record hr;
    EXPECT_TRUE(s.hit(ray(vec3(2, 0, 0), vec3(-1, 0, 0)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.u, 0.5); // equator point (1,0,0)
    EXPECT_NEAR(hr.v, 0.5);
    quad q(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), m);
    EXPECT_TRUE(q.hit(ray(vec3(0.25, 0.75, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.u, 0.25);
    EXPECT_NEAR(hr.v, 0.75);
    triangle t(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), m);
    EXPECT_TRUE(t.hit(ray(vec3(0.2, 0.2, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.u + hr.v, 0.4); // barycentric weights preserved
}
static void t_ppm_obj() {
    cur = "ppm";
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

    cur = "obj";
    std::string cube = (std::filesystem::temp_directory_path() / "rt_cube_test.obj").string();
    {
        std::ofstream f(cube);
        f << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n";
        f << "v 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n";
        f << "f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 5 1 4 8\n";
    }
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    std::vector<std::shared_ptr<triangle>> tris;
    EXPECT_TRUE(obj_loader::load_obj(cube, tris, m));
    EXPECT_TRUE((int)tris.size() == 12); // 6 quads fan-split
    EXPECT_TRUE(!obj_loader::load_obj(cube + ".missing", tris, m));
    // Loaded mesh actually intersects.
    hittable_list w;
    for (auto &t : tris)
        w.add(t);
    hit_record hr;
    EXPECT_TRUE(w.hit(ray(vec3(0.5, 0.5, 3), vec3(0, 0, -1)), 0.001, 1e30, hr));
}

static void t_film() {
    cur = "film";
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

int main() {
    t_vec3(); t_ray(); t_camera(); t_sphere(); t_list(); t_triangle_quad();
    t_materials(); t_sampler(); t_montecarlo(); t_onb_cosine(); t_emissive();
    t_defocus(); t_rr(); t_aabb(); t_bvh(); t_texture_uv(); t_ppm_obj(); t_film();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
