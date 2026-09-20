// Geometry + acceleration tests: shapes, list, AABB, BVH, SAH, hit UVs.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/aabb.h"
#include "geometry/hittable.h"
#include "geometry/sphere.h"
#include "geometry/triangle.h"
#include "geometry/quad.h"
#include "accel/bvh.h"
#include "material/material.h"

#include <memory>
#include <vector>

static void t_sphere() {
    test_current = "sphere";
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
    test_current = "list";
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
    test_current = "triquad";
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

static void t_aabb() {
    test_current = "aabb";
    aabb b(vec3(-1, -1, -1), vec3(1, 1, 1));
    EXPECT_TRUE(b.hit(ray(vec3(0, 0, 5), vec3(0, 0, -1)), 0.001, 1e30));
    EXPECT_TRUE(!b.hit(ray(vec3(0, 0, 5), vec3(0, 1, 0)), 0.001, 1e30));
    // Parallel ray inside slab survives; outside rejected.
    EXPECT_TRUE(b.hit(ray(vec3(0, 0, 0), vec3(1, 0, 0)), 0.001, 1e30));
    EXPECT_TRUE(!b.hit(ray(vec3(0, 5, 0), vec3(1, 0, 0)), 0.001, 1e30));
    aabb c(vec3(0, 0, 0), vec3(2, 2, 2));
    aabb u = aabb::surrounding(b, c);
    EXPECT_NEAR(u.minimum.x(), -1);
    EXPECT_NEAR(u.maximum.x(), 2);
}

static void t_bvh() {
    test_current = "bvh";
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

static void t_sah() {
    test_current = "sah";
    rng_seed(50);
    auto m = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));
    // 50 random spheres: SAH must agree with list everywhere.
    std::vector<std::shared_ptr<hittable>> objs;
    for (int i = 0; i < 50; ++i)
        objs.push_back(std::make_shared<sphere>(
            vec3(random_double(-5, 5), random_double(-2, 3), random_double(-6, 0)),
            random_double(0.1, 0.6), m));
    hittable_list list;
    for (auto &o : objs)
        list.add(o);
    std::vector<std::shared_ptr<hittable>> for_sah = objs;
    bvh_node sah(for_sah, 0, for_sah.size(), true);
    EXPECT_TRUE(sah.count_prims() == 50); // nothing lost in partition
    // Aim rays at random spheres so most hit (uniform rays mostly miss).
    std::vector<vec3> centers;
    for (auto &o : objs) {
        aabb b;
        o->bounding_box(b);
        centers.push_back((b.minimum + b.maximum) * 0.5);
    }
    int agree = 0;
    for (int k = 0; k < 200; ++k) {
        vec3 target = centers[(size_t)(random_double() * centers.size()) % centers.size()];
        vec3 o = target + vec3(random_double(-3, 3), random_double(-3, 3), random_double(2, 5));
        ray r(o, unit_vector(target - o)); // aimed, guaranteed near-miss-or-hit
        hit_record rl, rb;
        bool hl = list.hit(r, 0.001, 1e30, rl);
        bool hb = sah.hit(r, 0.001, 1e30, rb);
        EXPECT_TRUE(hl == hb);
        if (hl && hb) {
            EXPECT_TRUE(fabs(rl.t - rb.t) < 1e-9);
            ++agree;
        }
    }
    EXPECT_TRUE(agree > 150);
    // Two far clusters: SAH root splits (inner node), not a leaf.
    std::vector<std::shared_ptr<hittable>> cl;
    for (int i = 0; i < 6; ++i) {
        cl.push_back(std::make_shared<sphere>(vec3(-10 + 0.1 * i, 0, -5), 0.2, m));
        cl.push_back(std::make_shared<sphere>(vec3(10 + 0.1 * i, 0, -5), 0.2, m));
    }
    bvh_node clustered(cl, 0, cl.size(), true);
    EXPECT_TRUE(!clustered.is_leaf());
    EXPECT_TRUE(clustered.count_prims() == 12);
    // Degenerate: coincident boxes stay correct, no infinite split.
    std::vector<std::shared_ptr<hittable>> deg;
    for (int i = 0; i < 10; ++i)
        deg.push_back(std::make_shared<sphere>(vec3(0, 0, -3), 0.5, m));
    bvh_node dnode(deg, 0, deg.size(), true);
    EXPECT_TRUE(dnode.count_prims() == 10);
    hit_record hr;
    EXPECT_TRUE(dnode.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.t, 2.5);
}

static void t_uv() {    test_current = "uv";
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

static void t_smooth() {
    test_current = "smooth";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    // Tilted vertex normals on a z=0 tri: center hit blends to average.
    triangle t(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
               vec3(1, 0, 0), vec3(0, 1, 0), m);
    hit_record hr;
    // Ray at barycentric (u=0.25, v=0.25): point (0.25,0.25,0).
    EXPECT_TRUE(t.hit(ray(vec3(0.25, 0.25, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    vec3 expect = unit_vector(vec3(0, 0, 1) * 0.5 + vec3(1, 0, 0) * 0.25 +
                              vec3(0, 1, 0) * 0.25);
    EXPECT_NEAR(hr.normal.x(), expect.x());
    EXPECT_NEAR(hr.normal.y(), expect.y());
    EXPECT_NEAR(hr.normal.z(), expect.z());
    // Flat tri: face normal exactly (no blend drift).
    triangle f(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), m);
    EXPECT_TRUE(f.hit(ray(vec3(0.25, 0.25, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.normal.x(), 0);
    EXPECT_NEAR(hr.normal.y(), 0);
    EXPECT_NEAR(hr.normal.z(), 1);
    // Backface: smooth normal flips toward ray like flat does.
    EXPECT_TRUE(t.hit(ray(vec3(0.25, 0.25, -1), vec3(0, 0, 1)), 0.001, 1e30, hr));
    EXPECT_TRUE(hr.normal.z() < 0);
}

static void t_meshuv() {
    test_current = "meshuv";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    // Corner UVs (0,0),(1,0),(0,1): center hit must read (0.25,0.25).
    triangle t(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), 0.0, 0.0, 1.0, 0.0,
               0.0, 1.0, m);
    hit_record hr;
    EXPECT_TRUE(t.hit(ray(vec3(0.25, 0.25, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.u, 0.25);
    EXPECT_NEAR(hr.v, 0.25);
    EXPECT_TRUE(t.uv_present());
    // No-UV tri: barycentric fallback preserved.
    triangle f(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), m);
    EXPECT_TRUE(!f.uv_present());
    EXPECT_TRUE(f.hit(ray(vec3(0.25, 0.25, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.u + hr.v, 0.5);
}

void run_geometry_tests() {
    t_sphere();
    t_list();
    t_triangle_quad();
    t_aabb();
    t_bvh();
    t_sah();
    t_uv();
    t_smooth();
    t_meshuv();
}
