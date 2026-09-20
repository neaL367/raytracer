// Camera + scene-assembly tests: pinhole, defocus, OBJ loading.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/triangle.h"
#include "material/material.h"
#include "core/obj_loader.h"
#include "scene/scene.h"
#include "gpu/flatten.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static void t_camera() {
    test_current = "camera";
    vec3 u = unit_vector(camera().get_ray(0.5, 0.5).direction());
    EXPECT_NEAR(u.x(), 0); EXPECT_NEAR(u.y(), 0); EXPECT_NEAR(u.z(), -1);
}

static void t_defocus() {
    test_current = "defocus";
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

static void t_obj() {
    test_current = "obj";
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
    // vn faces: smooth normals load and shade tilted.
    std::string smooth =
        (std::filesystem::temp_directory_path() / "rt_smooth_test.obj").string();
    {
        std::ofstream f(smooth);
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        f << "vn 0 0 1\nvn 1 0 0\nvn 0 1 0\n";
        f << "f 1//1 2//2 3//3\n";
    }
    std::vector<std::shared_ptr<triangle>> stris;
    EXPECT_TRUE(obj_loader::load_obj(smooth, stris, m));
    EXPECT_TRUE((int)stris.size() == 1);
    EXPECT_TRUE(stris[0]->hit(ray(vec3(0.25, 0.25, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    // Blend of (0,0,1)*0.5 + (1,0,0)*0.25 + (0,1,0)*0.25, normalized.
    vec3 expect = unit_vector(vec3(0.25, 0.25, 0.5));
    EXPECT_NEAR(hr.normal.x(), expect.x());
    EXPECT_NEAR(hr.normal.z(), expect.z());
    // vt faces: UV flag on, interpolated UVs land inside the quad.
    std::string uvobj =
        (std::filesystem::temp_directory_path() / "rt_uv_test.obj").string();
    {
        std::ofstream f(uvobj);
        f << "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n";
        f << "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n";
        f << "f 1/1 2/2 3/3 4/4\n";
    }
    std::vector<std::shared_ptr<triangle>> uvtris;
    EXPECT_TRUE(obj_loader::load_obj(uvobj, uvtris, m));
    EXPECT_TRUE((int)uvtris.size() == 2);
    EXPECT_TRUE(uvtris[0]->uv_present());
    EXPECT_TRUE(uvtris[0]->hit(ray(vec3(0.5, 0.5, 1), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_TRUE(hr.u >= 0 && hr.u <= 1 && hr.v >= 0 && hr.v <= 1);
}

static void t_cornell() {
    test_current = "cornell";
    scene_data scene = build_cornell(16.0 / 9.0, 0.0);
    EXPECT_TRUE((int)scene.objs.size() == 18); // 5 walls + 12 box + 1 light
    EXPECT_TRUE((int)scene.lights.size() == 1);
    EXPECT_TRUE(scene.lights[0]->mat_ptr()->emitted().x() > 1); // bright
    hittable_list world;
    for (auto &o : scene.objs)
        world.add(o);
    hit_record hr;
    // Center ray: through (278,278) hits tall box front (z=295) first.
    EXPECT_TRUE(world.hit(scene.cam.get_ray(0.5, 0.5), 0.001, 1e30, hr));
    EXPECT_TRUE(hr.point.z() > 290 && hr.point.z() < 300);
    // Inside tall box looking +x: exits at x=430 wall.
    EXPECT_TRUE(world.hit(ray(vec3(300, 100, 350), vec3(1, 0, 0)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.point.x(), 430);
    // Default builder unchanged: ground + mesh/fallback + 2 spheres + light.
    scene_data def = build_default(16.0 / 9.0, 0.0);
    EXPECT_TRUE((int)def.lights.size() == 1);
    EXPECT_TRUE((int)def.objs.size() >= 16); // ground+12mesh+2sph+light
}

static void t_flatten() {
    test_current = "flatten";
    scene_data def = build_scene("default", 16.0 / 9.0, 0.0);
    flat_scene fs;
    EXPECT_TRUE(flatten_scene(def, fs));
    // Refs conserve prims; nodes form a real tree (or single leaf).
    EXPECT_TRUE((int)fs.refs.size() == (int)def.objs.size());
    EXPECT_TRUE(!fs.nodes.empty());
    int leaf_prims = 0;
    for (auto &n : fs.nodes)
        leaf_prims += (n.left < 0) ? n.count : 0;
    EXPECT_TRUE(leaf_prims == (int)def.objs.size());
    // Ground photo is image 0; cube photo2 is image 1 (registry).
    EXPECT_TRUE(!fs.gs.spheres.empty());
    EXPECT_TRUE(fs.gs.spheres[0].prm[0] == 5);
    EXPECT_TRUE(fabs(fs.gs.spheres[0].prm[1] - 0.0f) < 1e-6);
    EXPECT_TRUE((int)fs.images.size() == 2);
    EXPECT_TRUE(fs.images[0].w == 128 && fs.images[0].h == 64);
    EXPECT_TRUE(fs.images[1].w == 64 && fs.images[1].h == 64);
    bool cube_photo = false;
    for (auto &t : fs.gs.tris)
        if (t.prm[0] == 5 && fabs(t.prm[1] - 1.0f) < 1e-6)
            cube_photo = true;
    EXPECT_TRUE(cube_photo);
    // Dedupe: one texture on two spheres registers a single image.
    ppm_io::image tiny_px;
    tiny_px.w = 1;
    tiny_px.h = 1;
    tiny_px.px = {vec3(0.5, 0.5, 0.5)};
    auto shared_tex = std::make_shared<image_texture>(1, 1, tiny_px.px);
    auto shared_mat = std::make_shared<lambertian>(shared_tex);
    scene_data tiny;
    tiny.objs.push_back(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, shared_mat));
    tiny.objs.push_back(std::make_shared<sphere>(vec3(2, 0, -1), 0.5, shared_mat));
    flat_scene ft;
    EXPECT_TRUE(flatten_scene(tiny, ft));
    EXPECT_TRUE((int)ft.images.size() == 1); // shared ptr, one entry
    EXPECT_TRUE(ft.gs.spheres[0].prm[0] == 5 && ft.gs.spheres[1].prm[0] == 5);
    EXPECT_TRUE(fabs(ft.gs.spheres[1].prm[1] - 0.0f) < 1e-6); // same idx
    // Cornell: 18 quads flat, light first for NEE indexing.
    scene_data cor = build_scene("cornell", 1.0, 0.0);
    flat_scene fc;
    EXPECT_TRUE(flatten_scene(cor, fc));
    EXPECT_TRUE((int)fc.refs.size() == 18);
    EXPECT_TRUE(fc.nlights == 1);
    EXPECT_TRUE(fabs(fc.gs.quads[0].emit[0] - 7) < 1e-6); // light leads
    // Export values exact on a known material.
    auto m = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    float alb[4] = {}, alb2[4] = {}, emit[4] = {}, prm[4] = {};
    EXPECT_TRUE(m->export_gpu(alb, alb2, emit, prm));
    EXPECT_TRUE(fabs(alb[0] - 0.8) < 1e-6);
    EXPECT_TRUE(prm[0] == 1 && fabs(prm[1] - 0.3) < 1e-6);
}

void run_scene_tests() {
    t_camera();
    t_defocus();
    t_obj();
    t_cornell();
    t_flatten();
}
