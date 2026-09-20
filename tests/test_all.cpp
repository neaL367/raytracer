// Dependency-free tests. No Catch2 yet: zero-dep rule for M1,
// 30-line registry covers deterministic math checks.
#include "core/vec3.h"
#include "core/ray.h"
#include "camera/camera.h"
#include "geometry/sphere.h"

#include <cmath>
#include <cstdio>
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
    EXPECT_NEAR(dot(a, b), 1*4 + 2*(-1) + 3*0.5); // 3.5
    vec3 c = cross(vec3(1,0,0), vec3(0,1,0));
    EXPECT_NEAR(c.z(), 1);
    EXPECT_NEAR(unit_vector(vec3(0,3,4)).length(), 1);
}
static void t_ray() {
    cur = "ray";
    ray r(vec3(0,0,0), vec3(0,0,-1));
    vec3 p = r.at(0.5);
    EXPECT_NEAR(p.z(), -0.5);
}
static void t_camera() {
    cur = "camera";
    camera cam;
    vec3 d = cam.get_ray(0.5, 0.5).direction();
    vec3 u = unit_vector(d);
    EXPECT_NEAR(u.x(), 0); EXPECT_NEAR(u.y(), 0); EXPECT_NEAR(u.z(), -1);
}
static void t_sphere() {
    cur = "sphere";
    sphere s(vec3(0,0,-1), 0.5);
    hit_record rec;
    EXPECT_TRUE(s.hit(ray(vec3(0,0,0), vec3(0,0,-1)), 0.001, 1e30, rec));
    EXPECT_NEAR(rec.t, 0.5);
    EXPECT_NEAR(rec.normal.z(), 1); // front face +z
    EXPECT_TRUE(!s.hit(ray(vec3(0,0,0), vec3(0,1,0)), 0.001, 1e30, rec));
}

int main() {
    t_vec3(); t_ray(); t_camera(); t_sphere();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
