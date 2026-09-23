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
#include "geometry/volume.h"
#include "accel/bvh.h"
#include "accel/qbvh.h"
#include "accel/qbvh_flat.h"
#include "material/material.h"
#include "integrator/nesting.h"

#include <cmath>
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

static void t_motion() {
    test_current = "motion";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    sphere s(vec3(0, 0, -1), vec3(0, 1, -1), 0.0, 1.0, 0.5, m);
    // Lerp endpoints exact, midpoint exact.
    EXPECT_NEAR(s.center(0.0).y(), 0);
    EXPECT_NEAR(s.center(1.0).y(), 1);
    EXPECT_NEAR(s.center(0.5).y(), 0.5);
    EXPECT_NEAR(s.center(9.0).y(), 1);  // clamped past range
    EXPECT_NEAR(s.center(-9.0).y(), 0); // clamped before range
    // Hit follows time: t=0 front at z=-0.5, t=1 front at z=-0.5 shifted.
    hit_record r0, r1;
    EXPECT_TRUE(s.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1), 0.0), 0.001, 1e30, r0));
    EXPECT_TRUE(s.hit(ray(vec3(0, 1, 0), vec3(0, 0, -1), 1.0), 0.001, 1e30, r1));
    EXPECT_NEAR(r0.t, 0.5);
    EXPECT_NEAR(r1.t, 0.5);
    // Bounds union both endpoints.
    aabb b;
    EXPECT_TRUE(s.bounding_box(b));
    EXPECT_TRUE(b.minimum.y() <= -0.5 && b.maximum.y() >= 1.5);
    // Static sphere ignores time (same hit at any tm).
    sphere st(vec3(0, 0, -1), 0.5, m);
    hit_record ra, rb;
    EXPECT_TRUE(st.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1), 0.0), 0.001, 1e30, ra));
    EXPECT_TRUE(st.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1), 0.7), 0.001, 1e30, rb));
    EXPECT_NEAR(ra.t, rb.t);
}

static void t_volume() {
    test_current = "volume";
    rng_seed(90);
    auto phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
    auto border = std::make_shared<sphere>(vec3(0, 0, -1), 1.0, phase);
    constant_medium fog(border, 0.5, phase);
    // Through center: chord 2.0, scatter must land inside.
    hit_record hr;
    EXPECT_TRUE(fog.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_TRUE(hr.t > 0 && hr.t < 2.0);
    EXPECT_TRUE(hr.mat == phase);
    // Bounds == boundary bounds.
    aabb b;
    EXPECT_TRUE(fog.bounding_box(b));
    EXPECT_NEAR(b.minimum.z(), -2.0);
    EXPECT_NEAR(b.maximum.z(), 0.0);
    // Isotropic: unit direction, albedo passthrough.
    hit_record rec;
    rec.point = vec3(0, 0, 0);
    vec3 att;
    ray sc;
    EXPECT_TRUE(phase->scatter(ray(vec3(0, 0, 0), vec3(0, 0, -1)), rec, att, sc));
    EXPECT_NEAR(sc.direction().length(), 1);
    EXPECT_NEAR(att.x(), 0.9);
    // Thin ray missing the ball: no scatter.
    EXPECT_TRUE(!fog.hit(ray(vec3(5, 5, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
}

static void t_hetero() {
    test_current = "hetero";
    auto phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
    auto border = std::make_shared<sphere>(vec3(0, 0, -1), 1.0, phase);
    heterogeneous_medium het(border, 2.0, phase);
    // Modulation exact at sin zeros/peaks, bounded everywhere on grid.
    EXPECT_NEAR(het.modulation(vec3(0, 0, 0)), 0.5);
    double pk = 3.1415926535897932385 / 2.0;
    EXPECT_NEAR(het.modulation(vec3(pk / 5, pk / 4, pk / 6)), 1.0);
    EXPECT_NEAR(het.modulation(vec3(-pk / 5, pk / 4, pk / 6)), 0.0);
    for (int i = 0; i < 200; ++i) {
        double m = het.modulation(vec3(0.13 * i, -0.29 * i, 0.07 * i));
        EXPECT_TRUE(m >= 0 && m <= 1); // majorant exact: tracking unbiased
    }
    // Miss escapes; seeded hit is deterministic and inside the chord.
    hit_record hr;
    EXPECT_TRUE(!het.hit(ray(vec3(5, 5, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
    rng_seed(91);
    EXPECT_TRUE(het.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_TRUE(hr.t > 0 && hr.t < 2.0 && hr.mat == phase);
    rng_seed(91);
    hit_record hr2;
    EXPECT_TRUE(het.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1e30, hr2));
    EXPECT_NEAR(hr.t, hr2.t); // replay bit-exact
    EXPECT_NEAR(het.density_val(), 2.0);
    EXPECT_TRUE((het.freqs() - vec3(5, 4, 6)).length() < 1e-12);
    // Unbiasedness in situ: center ray sees constant m=0.5 (x=y=0 kills
    // the sines); escape fraction must match exp(-sig*0.5*L), L~=2.
    rng_seed(126);
    int esc = 0;
    const int ET = 20000;
    for (int i = 0; i < ET; ++i) {
        hit_record hre;
        if (!het.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1e30, hre))
            esc++;
    }
    double theory = std::exp(-2.0 * 0.5 * 1.999);
    EXPECT_TRUE(fabs((double)esc / ET - theory) < 0.01);
}

static void t_transmit() {
    test_current = "transmit";
    auto phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
    auto border = std::make_shared<sphere>(vec3(0, 0, -1), 1.0, phase);
    // Constant analytic: sigma 0.5 over chord 2 -> exp(-1).
    constant_medium fog(border, 0.5, phase);
    double exit = -1;
    ray through(vec3(0, 0, 1), vec3(0, 0, -1));
    EXPECT_NEAR(fog.transmittance(through, 0.001, 1e30, exit), std::exp(-1.0));
    EXPECT_TRUE(exit > 2.9 && exit < 3.1); // chord [1,3] on the exterior ray
    // Miss -> 1, clipped chord scales.
    EXPECT_NEAR(fog.transmittance(ray(vec3(5, 5, 0), vec3(0, 0, -1)), 0.001, 1e30, exit),
                1.0);
    EXPECT_NEAR(fog.transmittance(through, 0.5, 1.5, exit), std::exp(-0.25));
    // March: empty world -> 1; solid wall -> 0.
    hittable_list empty;
    std::vector<std::shared_ptr<hittable>> nomedia;
    EXPECT_NEAR(shadow_transmittance(empty, nomedia, vec3(0, 0, 0), vec3(0, 0, -1), 10.0,
                                     0.0),
                1.0);
    auto wall_mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hittable_list wall;
    wall.add(std::make_shared<quad>(vec3(-5, -5, -5), vec3(10, 0, 0), vec3(0, 10, 0),
                                    wall_mat));
    EXPECT_NEAR(shadow_transmittance(wall, nomedia, vec3(0, 0, 0), vec3(0, 0, -1), 10.0,
                                     0.0),
                0.0);
    // March through one medium == its analytic segment.
    hittable_list smoky;
    auto foggy = std::make_shared<constant_medium>(border, 0.5, phase);
    smoky.add(foggy);
    std::vector<std::shared_ptr<hittable>> onemedia = {foggy};
    EXPECT_NEAR(
        shadow_transmittance(smoky, onemedia, vec3(0, 0, 1), vec3(0, 0, -1), 10.0, 0.0),
        std::exp(-1.0));
    // Coincident twins multiply (densities add): exp(-2).
    auto foggy2 = std::make_shared<constant_medium>(border, 0.5, phase);
    smoky.add(foggy2);
    std::vector<std::shared_ptr<hittable>> twomedia = {foggy, foggy2};
    EXPECT_NEAR(
        shadow_transmittance(smoky, twomedia, vec3(0, 0, 1), vec3(0, 0, -1), 10.0, 0.0),
        std::exp(-2.0));
    // Interior solid blocks: opaque ball inside the fog on the shadow path
    // must give Tr 0 (M48: exit-advance skipped it, leaking Tr>0 whenever
    // the medium event came first — ~31% per ray here, so loop it).
    hittable_list blocked;
    auto foggy3 = std::make_shared<constant_medium>(border, 0.5, phase);
    blocked.add(foggy3);
    blocked.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.25, wall_mat));
    std::vector<std::shared_ptr<hittable>> blockedmedia = {foggy3};
    int leaks = 0;
    for (int k = 0; k < 50; ++k)
        if (shadow_transmittance(blocked, blockedmedia, vec3(0, 0, 1), vec3(0, 0, -1),
                                 10.0, 0.0) > 0.0)
            leaks++;
    EXPECT_TRUE(leaks == 0);
    // Scene-scale chord (r=2.5 ball, chord 5, sigma 0.3): march == analytic.
    auto big = std::make_shared<sphere>(vec3(0, 0, -1), 2.5, phase);
    auto bigfog = std::make_shared<constant_medium>(big, 0.3, phase);
    hittable_list bigworld;
    bigworld.add(bigfog);
    std::vector<std::shared_ptr<hittable>> bigmedia = {bigfog};
    // From inside (ball center) toward outside: exit at t=2.5.
    // dist 4 -> tmax ~4, chord [0.001, 2.5]: Tr = exp(-0.3*2.499).
    EXPECT_NEAR(shadow_transmittance(bigworld, bigmedia, vec3(0, 0, -1), vec3(0, 0, 1),
                                     4.0, 0.0),
                std::exp(-0.3 * 2.499));
    // Hetero ratio tracking is unbiased: mean over seeds ~= exp(-sig*M).
    heterogeneous_medium het(border, 2.0, phase);
    std::vector<std::shared_ptr<hittable>> hetmedia = {
        std::make_shared<heterogeneous_medium>(border, 2.0, phase)};
    hittable_list hworld;
    hworld.add(hetmedia[0]);
    double sum = 0;
    const int HT = 40;
    for (int s = 0; s < HT; ++s) {
        rng_seed(300 + (unsigned)s);
        sum += shadow_transmittance(hworld, hetmedia, vec3(0.3, 0.2, 0), vec3(0, 0, -1),
                                    10.0, 0.0);
    }
    // Offset ray chord ~1.866, mean modulation ~0.5 -> Tr ~= exp(-2*0.93).
    EXPECT_TRUE(sum / HT > 0.05 && sum / HT < 0.35);
}

static void t_trimotion() {
    test_current = "trimotion";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    // Static tri: vert_at ignores time, hit matches legacy endpoints.
    triangle st(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), m);
    EXPECT_TRUE((st.vert_at(0, 0.7) - vec3(0, 0, 0)).length() < 1e-12);
    hit_record hs;
    EXPECT_TRUE(st.hit(ray(vec3(0.2, 0.2, 1), vec3(0, 0, -1), 0.7), 0.001, 1e30, hs));
    EXPECT_NEAR(hs.t, 1.0);
    // Moving tri: z=0 plane at t=0, z=1 plane at t=1.
    triangle mt(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), m);
    mt.set_motion(vec3(0, 0, 1), vec3(1, 0, 1), vec3(0, 1, 1), 0.0, 1.0);
    EXPECT_TRUE((mt.vert_at(2, 0.0) - vec3(0, 1, 0)).length() < 1e-12);
    EXPECT_TRUE((mt.vert_at(2, 1.0) - vec3(0, 1, 1)).length() < 1e-12);
    EXPECT_TRUE((mt.vert_at(2, 0.5) - vec3(0, 1, 0.5)).length() < 1e-12);
    hit_record h0, h1;
    EXPECT_TRUE(mt.hit(ray(vec3(0.2, 0.2, 2), vec3(0, 0, -1), 0.0), 0.001, 1e30, h0));
    EXPECT_TRUE(mt.hit(ray(vec3(0.2, 0.2, 2), vec3(0, 0, -1), 1.0), 0.001, 1e30, h1));
    EXPECT_NEAR(h0.t, 2.0); // plane z=0 from z=2
    EXPECT_NEAR(h1.t, 1.0); // plane z=1 from z=2
    aabb b;
    EXPECT_TRUE(mt.bounding_box(b));
    EXPECT_TRUE(b.minimum.z() <= 0 && b.maximum.z() >= 1);
}

// fp32 replica of the GLSL modulation (mirrors common.glsl exactly).
static float het_m32(float x, float y, float z) {
    float v = 0.5f + 0.5f * sinf(5.0f * x) * sinf(4.0f * y) * sinf(6.0f * z);
    if (v < 0)
        v = 0;
    if (v > 1)
        v = 1;
    return v;
}

static void t_hetprec() {
    test_current = "hetprec";
    auto phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
    auto border = std::make_shared<sphere>(vec3(0, 0, -1), 1.0, phase);
    heterogeneous_medium het(border, 2.0, phase);
    // Modulation mean: fp64 real code vs fp32 GLSL replica agree.
    rng_seed(124);
    double s64 = 0, s32 = 0;
    const int N = 60000;
    for (int i = 0; i < N; ++i) {
        vec3 p(random_double(-2, 0), random_double(-2, 0), -1 + random_double(-1, 1));
        s64 += het.modulation(p);
        s32 += het_m32((float)p.x(), (float)p.y(), (float)p.z());
    }
    EXPECT_TRUE(fabs(s64 / N - 0.5) < 0.01); // symmetric modulation
    EXPECT_TRUE(fabs(s64 / N - s32 / N) < 1e-4); // fp32 mirror faithful
    // Full tracking loop in fp32 vs theory 1-exp(-sig*M) on an offset ray.
    double ox = 0.3, oy = 0.2;
    double t0 = 1 - std::sqrt(0.87), t1 = 1 + std::sqrt(0.87);
    double M = 0;
    const int Q = 2000;
    for (int i = 0; i < Q; ++i) {
        double t = t0 + (t1 - t0) * (i + 0.5) / Q;
        M += het.modulation(vec3(ox, oy, -t));
    }
    M *= (t1 - t0) / Q;
    rng_seed(125);
    int ev = 0;
    const int T = 60000;
    for (int i = 0; i < T; ++i) {
        float cursor = (float)t0;
        while (true) {
            float su = (float)random_double();
            if (su <= 0)
                su = 1e-7f;
            float s = -logf(su) / 2.0f;
            float x = cursor + s;
            if (x > (float)t1)
                break;
            float au = (float)random_double();
            if (het_m32((float)ox, (float)oy, (float)-x) > au) {
                ev++;
                break;
            }
            cursor = x;
        }
    }
    double theory = 1 - std::exp(-2.0 * M);
    EXPECT_TRUE(fabs((double)ev / T - theory) < 0.01); // fp32 tracking sound
}

static void t_qbvh() {
    test_current = "qbvh";
    rng_seed(31);
    auto m = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));
    std::vector<std::shared_ptr<hittable>> objs;
    objs.push_back(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, m));
    objs.push_back(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, m));
    objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, m));
    objs.push_back(std::make_shared<sphere>(vec3(1, 0.5, -2), 0.3, m));
    objs.push_back(std::make_shared<sphere>(vec3(0.5, -0.5, -1.5), 0.4, m));
    objs.push_back(std::make_shared<triangle>(vec3(-1, -1, -2), vec3(1, -1, -2),
                                              vec3(0, 1, -2), m));
    objs.push_back(std::make_shared<triangle>(vec3(0, 0, -3), vec3(1, 0, -3),
                                              vec3(0, 1, -3), m));
    objs.push_back(std::make_shared<quad>(vec3(-1, -1, -3), vec3(2, 0, 0),
                                          vec3(0, 2, 0), m));
    hittable_list list;
    for (auto &o : objs)
        list.add(o);
    bvh_node ref(objs, 0, objs.size());
    qbvh_node tree(ref); // same DFS order: bit-exact twin expected
    // Root box covers the scene.
    aabb b;
    EXPECT_TRUE(tree.bounding_box(b));
    EXPECT_TRUE(b.minimum.y() <= -100.5 && b.maximum.y() >= 1.0);
    int agree = 0;
    for (int k = 0; k < 400; ++k) {
        vec3 o(random_double(-2, 2), random_double(-2, 2), random_double(-1, 1));
        ray r(o, random_unit_vector());
        hit_record rl, rq, rb;
        bool hl = list.hit(r, 0.001, 1e30, rl);
        bool hb = ref.hit(r, 0.001, 1e30, rb);
        bool hq = tree.hit(r, 0.001, 1e30, rq);
        EXPECT_TRUE(hl == hb && hb == hq);
        if (hl && hb && hq) {
            EXPECT_TRUE(fabs(rl.t - rb.t) < 1e-9); // list agrees loosely
            // Bit-exact vs the binary twin: order, narrowing, ties.
            EXPECT_TRUE(rb.t == rq.t);
            EXPECT_TRUE(rb.mat == rq.mat);
            EXPECT_TRUE((rb.point - rq.point).length() == 0);
            ++agree;
        }
    }
    EXPECT_TRUE(agree > 100); // scene actually hit, not all misses
}

static void t_qflat() {
    test_current = "qflat";
    rng_seed(32);
    auto m = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));
    std::vector<std::shared_ptr<hittable>> objs;
    objs.push_back(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, m));
    objs.push_back(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, m));
    objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, m));
    objs.push_back(std::make_shared<sphere>(vec3(1, 0.5, -2), 0.3, m));
    objs.push_back(std::make_shared<sphere>(vec3(0.5, -0.5, -1.5), 0.4, m));
    objs.push_back(std::make_shared<triangle>(vec3(-1, -1, -2), vec3(1, -1, -2),
                                              vec3(0, 1, -2), m));
    objs.push_back(std::make_shared<triangle>(vec3(0, 0, -3), vec3(1, 0, -3),
                                              vec3(0, 1, -3), m));
    objs.push_back(std::make_shared<quad>(vec3(-1, -1, -3), vec3(2, 0, 0),
                                          vec3(0, 2, 0), m));
    hittable_list list;
    for (auto &o : objs)
        list.add(o);
    bvh_node ref(objs, 0, objs.size());
    std::vector<flat_qnode> flat;
    build_flat_qbvh(ref, flat);
    EXPECT_TRUE(!flat.empty());
    // Structure: slots bounded, leaves conserve prims exactly once.
    size_t leaf_prims = 0;
    size_t inner = 0;
    for (auto &qn : flat) {
        EXPECT_TRUE(qn.nslots >= 1 && qn.nslots <= 4);
        for (int s = 0; s < qn.nslots; ++s) {
            if (qn.slot[s].leaf)
                leaf_prims += qn.slot[s].prims.size();
            else {
                EXPECT_TRUE(qn.slot[s].node >= 0 &&
                            qn.slot[s].node < (int)flat.size());
                ++inner;
            }
        }
    }
    EXPECT_TRUE(leaf_prims == objs.size());
    // GPU pod: inverted boxes on dead slots, leaf flags line up.
    GPUQNode g0 = to_gpu_qnode(flat[0]);
    int live_leaves = 0;
    for (int s = 0; s < 4; ++s) {
        bool live = s < flat[0].nslots;
        if (live && flat[0].slot[s].leaf) {
            EXPECT_TRUE(g0.child[s] == -1 && g0.count[s] == 0);
            ++live_leaves;
        } else if (live) {
            EXPECT_TRUE(g0.child[s] >= 0);
        } else {
            EXPECT_TRUE(g0.bmin[s][0] > g0.bmax[s][0]); // inverted: never hit
            EXPECT_TRUE(g0.child[s] == -1 && g0.count[s] == 0);
        }
    }
    EXPECT_TRUE(live_leaves + (int)inner >= 1);
    // Traversal equivalence vs binary twin on seeded rays.
    int agree = 0;
    for (int k = 0; k < 400; ++k) {
        vec3 o(random_double(-2, 2), random_double(-2, 2), random_double(-1, 1));
        ray r(o, random_unit_vector());
        hit_record rb, rf;
        bool hb = ref.hit(r, 0.001, 1e30, rb);
        bool hf = flat_qbvh_hit(flat, r, 0.001, 1e30, rf);
        EXPECT_TRUE(hb == hf);
        if (hb && hf) {
            EXPECT_TRUE(rb.t == rf.t);
            EXPECT_TRUE(rb.mat == rf.mat);
            ++agree;
        }
    }
    EXPECT_TRUE(agree > 100);
}

static void t_tangents() {
    test_current = "tangents";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    hit_record hr;
    auto check_frame = [&](const hit_record &r) {
        EXPECT_TRUE(r.has_tangent);
        EXPECT_NEAR(r.tangent.length(), 1);
        EXPECT_TRUE(fabs(dot(r.tangent, r.normal)) < 1e-9);
    };
    // Sphere: hit off-pole carries a unit in-plane tangent.
    sphere sph(vec3(0, 0, -1), 0.5, m);
    EXPECT_TRUE(sph.hit(ray(vec3(0.3, 0.2, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
    check_frame(hr);
    // Quad: edge-u tangent, always valid.
    quad qd(vec3(-1, -1, -3), vec3(2, 0, 0), vec3(0, 2, 0), m);
    EXPECT_TRUE(qd.hit(ray(vec3(0, 0, 0), vec3(0, 0, -1)), 0.001, 1e30, hr));
    check_frame(hr);
    EXPECT_TRUE(fabs(hr.tangent.x() - 1.0) < 1e-9); // along +x edge
    // Triangle with corner UVs: derivative tangent in-plane.
    triangle tr(vec3(0, 0, 0), vec3(2, 0, 0), vec3(0, 2, 0), 0, 0, 1, 0, 0, 1, m);
    EXPECT_TRUE(tr.hit(ray(vec3(0.5, 0.5, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    check_frame(hr);
    // Barycentric fallback: e1 direction.
    triangle tb(vec3(0, 0, 0), vec3(2, 0, 0), vec3(0, 2, 0), m);
    EXPECT_TRUE(tb.hit(ray(vec3(0.5, 0.5, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    check_frame(hr);
    EXPECT_TRUE(fabs(hr.tangent.x() - 1.0) < 1e-9);
}

static void t_nesting() {
    test_current = "nesting";
    // Air-glass-milk-glass-air round trip through concentric shells.
    medium_stack st;
    EXPECT_TRUE(st.depth() == 1);
    EXPECT_NEAR(st.top().ior, 1.0);
    double eta = 0;
    int glass = 1, milk = 2, other = 3; // stand-in prim ids
    EXPECT_TRUE(st.resolve(1.5, 1, &glass, eta) == medium_stack::ENTER);
    EXPECT_NEAR(eta, 1.0 / 1.5);
    EXPECT_TRUE(st.depth() == 2);
    EXPECT_TRUE(st.resolve(1.33, 2, &milk, eta) == medium_stack::ENTER);
    EXPECT_NEAR(eta, 1.5 / 1.33);
    EXPECT_TRUE(st.depth() == 3);
    EXPECT_TRUE(st.top().pri == 2);
    // Exit inner first (identity), then outer: etas invert exactly.
    EXPECT_TRUE(st.resolve(1.33, 2, &milk, eta) == medium_stack::EXIT);
    EXPECT_NEAR(eta, 1.33 / 1.5);
    EXPECT_TRUE(st.depth() == 2);
    EXPECT_TRUE(st.resolve(1.5, 1, &glass, eta) == medium_stack::EXIT);
    EXPECT_NEAR(eta, 1.5 / 1.0);
    EXPECT_TRUE(st.depth() == 1);
    // Other ball while inside glass: same pri, different id -> PASS.
    medium_stack st2;
    EXPECT_TRUE(st2.resolve(1.5, 1, &glass, eta) == medium_stack::ENTER);
    EXPECT_TRUE(st2.resolve(1.5, 1, &other, eta) == medium_stack::PASS);
    EXPECT_TRUE(st2.depth() == 2); // untouched
    EXPECT_TRUE(st2.top().id == &glass);
    // Lower-pri dielectric from inside higher medium -> PASS.
    EXPECT_TRUE(st2.resolve(1.5, 0, &other, eta) == medium_stack::PASS);
    EXPECT_TRUE(st2.depth() == 2);
    // Overflow past CAP: 8th push refuses, stack intact.
    medium_stack st3;
    int ids[9] = {0};
    for (int k = 0; k < 7; ++k)
        EXPECT_TRUE(st3.resolve(1.5, k, &ids[k], eta) == medium_stack::ENTER);
    EXPECT_TRUE(st3.depth() == 8);
    EXPECT_TRUE(st3.resolve(1.5, 8, &ids[8], eta) == medium_stack::PASS);
    EXPECT_TRUE(st3.depth() == 8);
    // Pri-0 default reproduces legacy front_face etas (enter 1/ir, exit ir).
    medium_stack st4;
    int g0 = 7;
    EXPECT_TRUE(st4.resolve(1.5, 0, &g0, eta) == medium_stack::ENTER);
    EXPECT_NEAR(eta, 1.0 / 1.5);
    EXPECT_TRUE(st4.resolve(1.5, 0, &g0, eta) == medium_stack::EXIT);
    EXPECT_NEAR(eta, 1.5 / 1.0);
    // Material + record defaults.
    auto d = std::make_shared<dielectric>(1.5);
    EXPECT_TRUE(d->priority() == 0);
    EXPECT_NEAR(d->ior(), 1.5);
    auto dp = std::make_shared<dielectric>(1.5, 0.0, 2);
    EXPECT_TRUE(dp->priority() == 2);
    auto lam = std::make_shared<lambertian>(vec3(1, 1, 1));
    EXPECT_TRUE(lam->priority() == 0);
    hit_record rec;
    EXPECT_TRUE(!rec.nest_set);
    EXPECT_NEAR(rec.nest_eta, 0);
}

void run_geometry_tests() {
    t_sphere();
    t_list();
    t_triangle_quad();
    t_aabb();
    t_bvh();
    t_qbvh();
    t_qflat();
    t_sah();
    t_uv();
    t_smooth();
    t_meshuv();
    t_motion();
    t_trimotion();
    t_volume();
    t_hetero();
    t_hetprec();
    t_transmit();
    t_tangents();
    t_nesting();
}
