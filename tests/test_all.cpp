// Dependency-free tests. Same 30-line registry, deterministic seeds.
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/sphere.h"
#include "geometry/triangle.h"
#include "geometry/quad.h"
#include "material/material.h"

#include <cmath>
#include <cstdio>
#include <memory>

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

int main() {
    t_vec3(); t_ray(); t_camera(); t_sphere(); t_list(); t_triangle_quad();
    t_materials(); t_sampler(); t_montecarlo();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
