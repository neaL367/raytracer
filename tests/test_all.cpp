// Dependency-free unit tests. No Catch2/doctest: the project has no package
// manager wired up, and a 30-line registry covers what these tests need —
// deterministic math checks with no threading and no RNG past fixed seeds.
// Build with the default target; run with ctest or the binary directly.
#include "core/vec3.h"
#include "core/ray.h"
#include "core/aabb.h"
#include "core/sphere.h"
#include "core/triangle.h"
#include "core/quad.h"
#include "core/hittable_list.h"
#include "core/bvh.h"
#include "core/material.h"
#include "core/texture.h"
#include "core/onb.h"
#include "core/random.h"
#include "core/integrator.h"
#include "app/config.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace
{

struct test_case
{
    std::string name;
    void (*fn)();
};

inline std::vector<test_case> &registry()
{
    static std::vector<test_case> cases;
    return cases;
}

int checks = 0;
int check_failures = 0;
const char *current_test = "";

void check(bool cond, const char *expr, int line)
{
    ++checks;
    if (!cond)
    {
        ++check_failures;
        std::printf("  FAIL %s:%d: %s\n", current_test, line, expr);
    }
}

bool near(double a, double b, double eps = 1e-9)
{
    return std::fabs(a - b) <= eps;
}

} // namespace

#define TEST(name)                                            \
    static void test_body_##name();                           \
    static bool registered_##name = [] {                      \
        registry().push_back({#name, test_body_##name});      \
        return true;                                          \
    }();                                                      \
    static void test_body_##name()

#define EXPECT_TRUE(expr) check((expr), #expr, __LINE__)
#define EXPECT_NEAR(a, b) check(near((a), (b)), #a " ~= " #b, __LINE__)
#define EXPECT_NEAR_EPS(a, b, eps) check(near((a), (b), (eps)), #a " ~= " #b, __LINE__)

TEST(vec3_arithmetic)
{
    vec3 a(1, 2, 3);
    vec3 b(4, -1, 0.5);
    vec3 s = a + b;
    EXPECT_NEAR(s.x(), 5.0);
    EXPECT_NEAR(s.y(), 1.0);
    EXPECT_NEAR(s.z(), 3.5);
    vec3 d = a - b;
    EXPECT_NEAR(d.x(), -3.0);
    vec3 m = a * 2.0;
    EXPECT_NEAR(m.y(), 4.0);
    vec3 n = 0.5 * a;
    EXPECT_NEAR(n.x(), 0.5);
    vec3 c = a * b; // component-wise, not dot
    EXPECT_NEAR(c.x(), 4.0);
    EXPECT_NEAR(c.y(), -2.0);
    vec3 q = a / 2.0;
    EXPECT_NEAR(q.z(), 1.5);
    vec3 neg = -a;
    EXPECT_NEAR(neg.x(), -1.0);
    EXPECT_NEAR(dot(a, b), 1 * 4 + 2 * -1 + 3 * 0.5);
    vec3 cr = cross(vec3(1, 0, 0), vec3(0, 1, 0));
    EXPECT_NEAR(cr.x(), 0.0);
    EXPECT_NEAR(cr.y(), 0.0);
    EXPECT_NEAR(cr.z(), 1.0);
    EXPECT_NEAR(vec3(3, 4, 0).length(), 5.0);
    vec3 u = unit_vector(vec3(0, 0, 5));
    EXPECT_NEAR(u.z(), 1.0);
    EXPECT_NEAR(a[0], 1.0);
    EXPECT_NEAR(a[2], 3.0);
}

TEST(ray_at)
{
    ray r(vec3(1, 2, 3), vec3(0, 1, 0));
    vec3 p = r.at(2.5);
    EXPECT_NEAR(p.x(), 1.0);
    EXPECT_NEAR(p.y(), 4.5);
    EXPECT_NEAR(p.z(), 3.0);
    // Precomputed inverse matches per-component division, including inf.
    EXPECT_NEAR(r.inv_direction().y(), 1.0);
    ray px(vec3(0, 0, 0), vec3(0, 2, 0));
    EXPECT_TRUE(std::isinf(px.inv_direction().x()));
}

TEST(sphere_hit)
{
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    sphere s(vec3(0, 0, -1), 0.5, mat);
    hit_record rec;
    EXPECT_TRUE(s.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1000.0, rec));
    EXPECT_NEAR(rec.t, 0.5);
    EXPECT_NEAR(rec.point.z(), -0.5);
    EXPECT_TRUE(rec.front_face);
    EXPECT_NEAR(rec.normal.z(), 1.0);
    // Miss above the sphere.
    EXPECT_TRUE(!s.hit(ray(vec3(0, 0, 0), vec3(0, 1, 0)), 0.001, 1000.0, rec));
    // Inside-out hit flips the normal.
    EXPECT_TRUE(s.hit(ray(vec3(0, 0, -1), vec3(0, 0, 1)), 0.001, 1000.0, rec));
    EXPECT_TRUE(!rec.front_face);
}

TEST(aabb_slab)
{
    aabb box(vec3(-1, -1, -1), vec3(1, 1, 1));
    EXPECT_TRUE(box.hit(ray(vec3(0, 0, 5), vec3(0, 0, -1)), 0.001, 1000.0));
    EXPECT_TRUE(!box.hit(ray(vec3(0, 0, 5), vec3(0, 0, 1)), 0.001, 1000.0));
    // Parallel to x/y slabs, moving in -z: infinities must not reject.
    EXPECT_TRUE(box.hit(ray(vec3(0, 0, 5), vec3(0, 0, -1)), 0.001, 1000.0));
    // Parallel and outside the slab: must reject.
    EXPECT_TRUE(!box.hit(ray(vec3(5, 5, 5), vec3(0, 0, -1)), 0.001, 1000.0));
    EXPECT_NEAR(box.surface_area(), 24.0);
}

TEST(quad_hit)
{
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    quad q(vec3(-1, 0, -2), vec3(2, 0, 0), vec3(0, 2, 0), mat);
    hit_record rec;
    EXPECT_TRUE(q.hit(ray(vec3(0, 1, 0), vec3(0, 0, -1)), 0.001, 1000.0, rec));
    EXPECT_NEAR(rec.t, 2.0);
    // Outside the [0,1] barycentric range: miss.
    EXPECT_TRUE(!q.hit(ray(vec3(5, 5, 0), vec3(0, 0, -1)), 0.001, 1000.0, rec));
    EXPECT_NEAR(q.area(), 4.0);
    // Uniform sample stays inside the quad bounds.
    vec3 p = q.sample();
    EXPECT_TRUE(p.x() >= -1.0 && p.x() <= 1.0 && p.y() >= 0.0 && p.y() <= 2.0);
}

TEST(triangle_hit)
{
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    triangle t(vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2), mat);
    hit_record rec;
    EXPECT_TRUE(t.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1000.0, rec));
    EXPECT_NEAR(rec.t, 2.0);
    EXPECT_TRUE(!t.hit(ray(vec3(0, 0, 5), vec3(0, 0, 1)), 0.001, 1000.0, rec));
    aabb b;
    EXPECT_TRUE(t.bounding_box(b));
    EXPECT_TRUE(b.min().x() < b.max().x() && b.min().y() < b.max().y());
}

TEST(checker_parity)
{
    checker_texture c(1.0, vec3(1, 1, 1), vec3(0, 0, 0));
    // sin products: (+,+,+) cell is even, one flipped sign is odd.
    vec3 even = c.value(0, 0, vec3(0.5, 0.5, 0.5));
    EXPECT_NEAR(even.x(), 1.0);
    vec3 odd = c.value(0, 0, vec3(-0.5, 0.5, 0.5));
    EXPECT_NEAR(odd.x(), 0.0);
    solid_color s(vec3(0.2, 0.4, 0.6));
    vec3 v = s.value(0.3, 0.7, vec3(9, 9, 9));
    EXPECT_NEAR(v.y(), 0.4);
    EXPECT_TRUE(!s.needs_uv() && !c.needs_uv());
}

TEST(image_texture_missing)
{
    image_texture missing("assets/does-not-exist.ppm");
    vec3 v = missing.value(0.5, 0.5, vec3(0, 0, 0));
    EXPECT_NEAR(v.x(), 1.0); // magenta fallback
    EXPECT_NEAR(v.y(), 0.0);
    EXPECT_NEAR(v.z(), 1.0);
}

TEST(image_texture_load)
{
    // Calibration asset: top-left corner is inside the white border.
    image_texture tex("assets/uv_check.ppm");
    EXPECT_TRUE(tex.needs_uv());
    vec3 corner = tex.value(0.0, 1.0, vec3(0, 0, 0));
    EXPECT_NEAR_EPS(corner.x(), 1.0, 1e-2);
    EXPECT_NEAR_EPS(corner.y(), 1.0, 1e-2);
}

TEST(onb_orthonormal)
{
    onb basis(vec3(0, 0, 1));
    vec3 d(0.3, 0.4, 0.5);
    vec3 w = basis.local(d.x(), d.y(), d.z());
    EXPECT_NEAR(w.length(), d.length());
    // Axis-aligned normal keeps z dominant.
    EXPECT_TRUE(w.z() > std::fabs(w.x()) && w.z() > std::fabs(w.y()));
}

TEST(lambertian_pdf)
{
    lambertian lamb(vec3(0.8, 0.8, 0.8));
    hit_record rec;
    rec.normal = vec3(0, 1, 0);
    rec.point = vec3(0, 0, 0);
    // Normal incidence: cos/PI.
    double pdf = lamb.scattering_pdf(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec,
                                     ray(vec3(0, 0, 0), vec3(0, 1, 0)));
    EXPECT_NEAR(pdf, 1.0 / std::numbers::pi);
    // Below the surface: zero.
    double below = lamb.scattering_pdf(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec,
                                       ray(vec3(0, 0, 0), vec3(0, -1, 0)));
    EXPECT_NEAR(below, 0.0);
    EXPECT_TRUE(!lamb.specular() && !lamb.needs_uv());
}

TEST(materials_specular)
{
    metal m(vec3(0.8, 0.8, 0.8));
    dielectric glass(1.5);
    diffuse_light light(vec3(3, 3, 3));
    EXPECT_TRUE(m.specular() && glass.specular());
    EXPECT_NEAR(light.emitted().x(), 3.0);
    ray scattered;
    vec3 attenuation;
    hit_record rec;
    rec.normal = vec3(0, 1, 0);
    rec.point = vec3(0, 0, 0);
    EXPECT_TRUE(!light.scatter(ray(vec3(0, 1, 0), vec3(0, -1, 0)), rec, attenuation, scattered));
}

TEST(bvh_matches_flat_list)
{
    // BVH must return the same closest hit as a flat scan, ray for ray.
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hittable_list list;
    list.add(std::make_shared<sphere>(vec3(0, 0, -2), 0.5, mat));
    list.add(std::make_shared<sphere>(vec3(1.5, 0, -3), 0.7, mat));
    list.add(std::make_shared<sphere>(vec3(-1.5, 0.5, -2.5), 0.4, mat));
    bvh_node root(list, 2);

    set_deterministic_rng(true, 7u);
    reseed_thread_rng(7u); // pin the stream: earlier tests may have created it from random_device
    for (int k = 0; k < 200; ++k)
    {
        vec3 o(random_double(-3, 3), random_double(-3, 3), 0);
        ray r(o, vec3(0, 0, -1));
        hit_record a, b;
        bool ha = list.hit(r, 0.001, 1000.0, a);
        bool hb = root.hit(r, 0.001, 1000.0, b);
        EXPECT_TRUE(ha == hb);
        if (ha && hb)
            EXPECT_NEAR_EPS(a.t, b.t, 1e-9);
    }
    set_deterministic_rng(false);

    size_t n1 = 0, l1 = 0, d1 = 0, n2 = 0, l2 = 0, d2 = 0;
    root.census(n1, l1, d1);
    bvh_node root2(list, 2);
    root2.census(n2, l2, d2);
    EXPECT_TRUE(n1 == n2 && l1 == l2 && d1 == d2); // deterministic build
}

TEST(deterministic_rng)
{
    set_deterministic_rng(true, 123u);
    reseed_thread_rng(123u);
    double a1 = random_double();
    double a2 = random_double();
    reseed_thread_rng(123u);
    EXPECT_NEAR(random_double(), a1);
    EXPECT_NEAR(random_double(), a2);
    set_deterministic_rng(false);
}

TEST(bvh_flatten_covers_all)
{
    // Every primitive reachable exactly once through the flat arrays, with
    // node/leaf counts agreeing with census.
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hittable_list list;
    auto s0 = std::make_shared<sphere>(vec3(0, 0, -2), 0.5, mat);
    auto s1 = std::make_shared<sphere>(vec3(1.5, 0, -3), 0.7, mat);
    auto t0 = std::make_shared<triangle>(vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2), mat);
    list.add(s0);
    list.add(s1);
    list.add(t0);
    bvh_node root(list, 2);

    std::map<const hittable *, FlatLeafRef> ids;
    int next_sphere = 0, next_tri = 0;
    for (const auto &o : list.objects_ref())
    {
        FlatLeafRef ref;
        if (dynamic_cast<const sphere *>(o.get()))
        {
            ref.type = 0;
            ref.index = next_sphere++;
        }
        else
        {
            ref.type = 1;
            ref.index = next_tri++;
        }
        ids[o.get()] = ref;
    }

    std::vector<FlatNode> nodes;
    std::vector<FlatLeafRef> refs;
    root.flatten(nodes, refs, [&](const std::shared_ptr<hittable> &p) { return ids[p.get()]; });

    size_t cn = 0, cl = 0, cd = 0;
    root.census(cn, cl, cd);
    EXPECT_TRUE(nodes.size() == cn);
    EXPECT_TRUE(refs.size() == list.size());

    // Walk the flat links from the root: visit every node once, collect refs.
    std::vector<int> stack = {0};
    std::vector<bool> seen(nodes.size(), false);
    size_t leaf_nodes = 0, ref_total = 0;
    while (!stack.empty())
    {
        int ni = stack.back();
        stack.pop_back();
        EXPECT_TRUE(ni >= 0 && static_cast<size_t>(ni) < nodes.size());
        EXPECT_TRUE(!seen[ni]);
        seen[ni] = true;
        const FlatNode &n = nodes[ni];
        if (n.left < 0)
        {
            ++leaf_nodes;
            ref_total += n.count;
            for (int k = 0; k < n.count; ++k)
            {
                FlatLeafRef r = refs[n.start + k];
                EXPECT_TRUE(r.type == 0 || r.type == 1);
                EXPECT_TRUE(r.index >= 0);
            }
        }
        else
        {
            stack.push_back(n.left);
            stack.push_back(n.right);
        }
    }
    for (bool s : seen)
        EXPECT_TRUE(s);
    EXPECT_TRUE(leaf_nodes == cl);
    EXPECT_TRUE(ref_total == list.size());
    // Root box matches the flat list's own bounds.
    aabb list_box, root_box;
    EXPECT_TRUE(list.bounding_box(list_box));
    EXPECT_TRUE(root.bounding_box(root_box));
    EXPECT_NEAR(list_box.min().x(), root_box.min().x());
    EXPECT_NEAR(list_box.max().z(), root_box.max().z());
}

TEST(normal_integrator_shade)
{
    // Up-facing normal maps to (0.5, 1.0, 0.5); miss maps to sky, same as
    // the GPU kernel's else-branch must produce.
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hittable_list world;
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, mat));
    normal_integrator shade;
    std::vector<std::shared_ptr<quad>> no_lights;
    // Straight down onto the sphere top: normal (0,1,0) -> (0.5,1.0,0.5).
    vec3 c = shade.Li(ray(vec3(0, 2, -1), vec3(0, -1, 0)), world, no_lights, 50);
    EXPECT_NEAR(c.x(), 0.5);
    EXPECT_NEAR(c.y(), 1.0);
    EXPECT_NEAR(c.z(), 0.5);
    vec3 sky = shade.Li(ray(vec3(0, 0, 0), vec3(0, 0, 1)), world, no_lights, 50);
    EXPECT_NEAR(sky.x(), 0.75); // horizon: half white, half (0.5,0.7,1.0)
    EXPECT_NEAR(sky.y(), 0.85);
    EXPECT_NEAR(sky.z(), 1.0);
}

TEST(cli_shade_mode)
{
    // parse_cli takes argv-style char**; string literals need the cast.
    char *ok_argv[] = {const_cast<char *>("prog"), const_cast<char *>("--shade"),
                       const_cast<char *>("normal")};
    render_config cfg;
    EXPECT_TRUE(parse_cli(3, ok_argv, cfg) == cli_result::run);
    EXPECT_TRUE(cfg.shade_mode == "normal");
    char *bad_argv[] = {const_cast<char *>("prog"), const_cast<char *>("--shade"),
                        const_cast<char *>("phong")};
    render_config cfg2;
    EXPECT_TRUE(parse_cli(3, bad_argv, cfg2) == cli_result::error);
    char *unknown_argv[] = {const_cast<char *>("prog"), const_cast<char *>("--frobnicate")};
    render_config cfg3;
    EXPECT_TRUE(parse_cli(2, unknown_argv, cfg3) == cli_result::error);
}

TEST(shadow_transmittance)
{
    // Empty world: pure Beer-Lambert, exact.
    hittable_list empty;
    EXPECT_NEAR(shadow_transmittance(empty, 0.1, vec3(0, 0, 0), vec3(1, 0, 0), 10.0),
                std::exp(-1.0));
    EXPECT_NEAR(shadow_transmittance(empty, 0.0, vec3(0, 0, 0), vec3(1, 0, 0), 10.0), 1.0);
    // Blocked shadow ray reads 0 at any density.
    auto mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hittable_list world;
    world.add(std::make_shared<sphere>(vec3(5, 0, 0), 1.0, mat));
    EXPECT_NEAR(shadow_transmittance(world, 0.1, vec3(0, 0, 0), vec3(1, 0, 0), 10.0), 0.0);
    // Clear of the sphere: extinction only.
    EXPECT_NEAR(shadow_transmittance(world, 0.1, vec3(0, 0, 0), vec3(0, 1, 0), 10.0),
                std::exp(-1.0));
}

int main()
{
    int test_failures = 0;
    for (const auto &t : registry())
    {
        current_test = t.name.c_str();
        int before = check_failures;
        t.fn();
        if (check_failures == before)
            std::printf("PASS %s\n", t.name.c_str());
        else
            ++test_failures;
    }
    std::printf("%d tests, %d checks, %d failed\n",
                static_cast<int>(registry().size()), checks, test_failures);
    return test_failures == 0 ? 0 : 1;
}
