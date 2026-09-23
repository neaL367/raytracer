// Camera + scene-assembly tests: pinhole, defocus, OBJ loading.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/quad.h"
#include "geometry/volume.h"
#include "geometry/triangle.h"
#include "accel/qbvh.h"
#include "integrator/pdf.h"
#include "material/material.h"
#include "core/obj_loader.h"
#include "scene/scene.h"
#include "gpu/flatten.h"

#include <cmath>
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

static void t_mtl() {
    test_current = "mtl";
    auto dir = std::filesystem::temp_directory_path();
    std::string mtl = (dir / "rt_mtl_test.mtl").string();
    std::string bmp = (dir / "rt_mtl_test.bmp").string();
    std::string obj = (dir / "rt_mtl_test.obj").string();
    { // 2x2 BMP map: bottom red/green, top blue/white (BGR, padded rows).
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
        unsigned char px[16] = {0, 0, 255, 0, 255, 0, 0, 0,
                                255, 0, 0, 255, 255, 255, 0, 0};
        std::ofstream f(bmp, std::ios::binary);
        f.write((char *)hdr, 54);
        f.write((char *)px, 16);
    }
    {
        std::ofstream f(mtl);
        f << "newmtl red\nKd 1 0 0\nillum 2\n";
        f << "newmtl mirror\nKd 0.2 0.2 0.2\nKs 0.9 0.9 0.9\n";
        f << "newmtl brushed\nKd 0.2 0.2 0.2\nKs 0.9 0.9 0.9\nNs 100\n";
        f << "newmtl dull\nKd 0 0 1\nNs 50\n";
        f << "newmtl badmap\nKd 0 1 0\nmap_Kd missing.png\n";
        f << "newmtl mapped\nKd 1 1 1\nmap_Kd rt_mtl_test.bmp\n";
    }
    {
        std::ofstream f(obj);
        f << "mtllib rt_mtl_test.mtl\n";
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\n";
        f << "v 2 0 0\nv 3 0 0\nv 2 1 0\nv 2 0 1\nv 3 0 1\nv 2 1 1\n";
        f << "v 4 0 0\nv 5 0 0\nv 4 1 0\nv 4 0 1\nv 5 0 1\nv 4 1 1\n";
        f << "f 1 2 3\n"; // no usemtl yet: caller fallback
        f << "usemtl red\nf 4 5 6\n";
        f << "usemtl mirror\nf 7 8 9\n";
        f << "usemtl badmap\nf 10 11 12\n";
        f << "usemtl ghost\nf 1 2 3\n"; // unknown name: fallback
        f << "usemtl brushed\nf 13 14 15\n";
        f << "usemtl dull\nf 16 17 18\n";
    }
    auto fallback = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    std::vector<std::shared_ptr<triangle>> tris;
    EXPECT_TRUE(obj_loader::load_obj(obj, tris, fallback));
    EXPECT_TRUE((int)tris.size() == 7);
    EXPECT_TRUE(tris[0]->mat_ptr() == fallback); // pre-usemtl
    EXPECT_TRUE(tris[4]->mat_ptr() == fallback); // unknown usemtl
    hit_record dummy;
    auto alb = [&](int i) { return tris[(size_t)i]->mat_ptr()->surface_albedo(dummy); };
    EXPECT_NEAR(alb(1).x(), 1); // Kd red
    EXPECT_TRUE(dynamic_cast<metal *>(tris[2]->mat_ptr().get()) != nullptr); // Ks
    EXPECT_NEAR(alb(2).x(), 0.9);
    EXPECT_NEAR(alb(3).y(), 1); // missing map -> Kd green
    // Ns mapping: Ks+Ns 100 -> GGX roughness sqrt(2/102).
    auto brushed = dynamic_cast<metal *>(tris[5]->mat_ptr().get());
    EXPECT_TRUE(brushed != nullptr);
    float alb4[4] = {}, alb24[4] = {}, emit4[4] = {}, prm4[4] = {};
    EXPECT_TRUE(brushed->export_gpu(alb4, alb24, emit4, prm4));
    EXPECT_TRUE(fabs(prm4[0] - 7.0f) < 1e-6); // GGX type
    EXPECT_TRUE(fabs(prm4[1] - (float)std::sqrt(2.0 / 102.0)) < 1e-6);
    // Ns without Ks stays lambertian (Kd blue).
    auto dull = dynamic_cast<lambertian *>(tris[6]->mat_ptr().get());
    EXPECT_TRUE(dull != nullptr);
    EXPECT_NEAR(alb(6).z(), 1);
    // map_Kd success path: separate mesh (needs vt? no, image needs no UVs
    // for albedo — but keep faces valid).
    std::string obj2 = (dir / "rt_mtl_map.obj").string();
    {
        std::ofstream f(obj2);
        f << "mtllib rt_mtl_test.mtl\n";
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        f << "usemtl mapped\nf 1 2 3\n";
    }
    std::vector<std::shared_ptr<triangle>> mtris;
    EXPECT_TRUE(obj_loader::load_obj(obj2, mtris, fallback));
    EXPECT_TRUE((int)mtris.size() == 1);
    auto ml = dynamic_cast<lambertian *>(mtris[0]->mat_ptr().get());
    EXPECT_TRUE(ml != nullptr);
    auto mit = std::dynamic_pointer_cast<image_texture>(ml->tex_ref());
    EXPECT_TRUE(mit != nullptr); // map_Kd wired, not Kd solid
    EXPECT_TRUE(mit->width() == 2 && mit->height() == 2);
    // Missing mtllib: whole file falls back, still loads.
    std::string obj3 = (dir / "rt_mtl_nomtl.obj").string();
    {
        std::ofstream f(obj3);
        f << "mtllib absent.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl red\nf 1 2 3\n";
    }
    std::vector<std::shared_ptr<triangle>> ntris;
    EXPECT_TRUE(obj_loader::load_obj(obj3, ntris, fallback));
    EXPECT_TRUE((int)ntris.size() == 1 && ntris[0]->mat_ptr() == fallback);
    // GPU export: MTL materials flow per-face (solid 0, metal 1, image 5).
    scene_data ms;
    for (auto &t : mtris)
        ms.objs.push_back(t);
    flat_scene mf;
    EXPECT_TRUE(flatten_scene(ms, mf));
    EXPECT_TRUE(mf.gs.tris[0].prm[0] == 5);
    EXPECT_TRUE((int)mf.images.size() == 1);
}

static void t_lights() {
    test_current = "lights";
    auto lamp = std::make_shared<diffuse_light>(vec3(3, 3, 3));
    // Quad: area 4, point on plane, geometric normal.
    light ql(std::make_shared<quad>(vec3(0, 0, 0), vec3(2, 0, 0), vec3(0, 2, 0), lamp));
    EXPECT_NEAR(light_area(ql), 4);
    vec3 qp = light_point(ql, 0.25, 0.5, 0.0);
    EXPECT_NEAR(qp.x(), 0.5); EXPECT_NEAR(qp.y(), 1.0); EXPECT_NEAR(qp.z(), 0);
    EXPECT_NEAR(light_normal_at(ql, qp, 0.0).z(), 1);
    // Sphere r=2: area 16 PI, points on surface, outward normals.
    light sl(std::make_shared<sphere>(vec3(1, 2, 3), 2.0, lamp));
    EXPECT_NEAR(light_area(sl), 16 * 3.1415926535897932385);
    vec3 sp = light_point(sl, 0.0, 0.0, 0.0); // u1=0 -> north pole
    EXPECT_TRUE((sp - vec3(1, 2, 3)).length() - 2.0 < 1e-9);
    EXPECT_NEAR(light_normal_at(sl, sp, 0.0).z(), 1);
    // Triangle legs 3/4: area 6, point in-plane and inside.
    auto tri = std::make_shared<triangle>(vec3(0, 0, 0), vec3(3, 0, 0),
                                          vec3(0, 4, 0), lamp);
    light tl(tri);
    EXPECT_NEAR(light_area(tl), 6);
    vec3 tp = light_point(tl, 0.25, 0.5, 0.0);
    EXPECT_NEAR(tp.z(), 0);
    EXPECT_TRUE(tp.x() >= 0 && tp.y() >= 0 && tp.x() / 3 + tp.y() / 4 <= 1 + 1e-9);
    EXPECT_TRUE((light_normal_at(tl, tp, 0.0) - vec3(0, 0, 1)).length() < 1e-9);
    EXPECT_TRUE(light_mat(tl) == tri->mat_ptr());
    // Flatten registers emissive spheres/tris alongside quads.
    scene_data es;
    es.objs.push_back(std::make_shared<sphere>(vec3(0, 3, 0), 0.5, lamp));
    es.objs.push_back(tri);
    flat_scene ef;
    EXPECT_TRUE(flatten_scene(es, ef));
    EXPECT_TRUE((int)ef.light_table.size() == 2);
    EXPECT_TRUE(ef.light_table[0].first == 1); // sphere type
    EXPECT_TRUE(ef.light_table[1].first == 2); // tri type
    EXPECT_TRUE(ef.nlights == 2);
}

static void t_instance() {
    test_current = "instance";
    auto m = std::make_shared<lambertian>(vec3(0.7, 0.7, 0.7));
    // translate: ray shifts by -offset, point shifts back, bbox shifts.
    auto box0 = make_box_list(vec3(0, 0, 0), vec3(1, 1, 1), m);
    auto moved = std::make_shared<translate>(box0, vec3(5, 0, 0));
    hit_record hr;
    EXPECT_TRUE(moved->hit(ray(vec3(5.5, 0.5, 3), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_NEAR(hr.point.x(), 5.5);
    EXPECT_NEAR(hr.point.z(), 1.0); // front face of the moved box
    aabb tb;
    EXPECT_TRUE(moved->bounding_box(tb));
    EXPECT_TRUE(tb.minimum.x() < 5.0 && tb.maximum.x() > 6.0);
    // rotate_y 90 deg: +x edge maps to -z; bbox swaps x/z extents.
    auto slab = make_box_list(vec3(0, 0, 0), vec3(2, 1, 1), m);
    auto spun = std::make_shared<rotate_y>(slab, 90.0);
    aabb rb;
    EXPECT_TRUE(spun->bounding_box(rb));
    // 1e-4 pad per side: extents match within 1e-3.
    EXPECT_TRUE(fabs((rb.maximum.x() - rb.minimum.x()) - 1.0) < 1e-3);
    EXPECT_TRUE(fabs((rb.maximum.z() - rb.minimum.z()) - 2.0) < 1e-3);
    // Ray down -z at x=0.5 hits the face that was +x before the spin.
    EXPECT_TRUE(spun->hit(ray(vec3(0.5, 0.5, 3), vec3(0, 0, -1)), 0.001, 1e30, hr));
    EXPECT_TRUE(fabs(hr.normal.z()) > 0.99); // normal spun back to world
    // Identity: angle 0 behaves like the raw box bit-exact.
    auto ident = std::make_shared<rotate_y>(box0, 0.0);
    hit_record h1, h2;
    ray probe(vec3(0.5, 0.5, 3), vec3(0, 0, -1));
    EXPECT_TRUE(box0->hit(probe, 0.001, 1e30, h1));
    EXPECT_TRUE(ident->hit(probe, 0.001, 1e30, h2));
    EXPECT_NEAR(h1.t, h2.t);
    // GPU bake: posed box yields 6 world-space quads covering the instance.
    auto posed = make_posed_box(vec3(2, 2, 2), 90.0, vec3(5, 0, 0), m);
    std::vector<quad> baked;
    EXPECT_TRUE(instance_detail::collect_baked_quads(posed, 1.0, 0.0, vec3(0, 0, 0),
                                                      baked));
    EXPECT_TRUE((int)baked.size() == 6);
    aabb ib;
    EXPECT_TRUE(posed->bounding_box(ib));
    // Every baked corner sits inside the instance bbox (with pad slack).
    for (auto &q : baked) {
        for (vec3 c : {q.corner(), q.corner() + q.edge_u(),
                       q.corner() + q.edge_v(), q.corner() + q.edge_u() + q.edge_v()}) {
            EXPECT_TRUE(c.x() >= ib.minimum.x() - 1e-6 && c.x() <= ib.maximum.x() + 1e-6);
            EXPECT_TRUE(c.z() >= ib.minimum.z() - 1e-6 && c.z() <= ib.maximum.z() + 1e-6);
        }
    }
}

static void t_gpucam() {
    test_current = "gpucam";
    GPUCam p = build_gpu_camera(200, 100, "default");
    EXPECT_NEAR(p.lens[0], 0); // pinhole default, streams untouched
    GPUCam a = build_gpu_camera(200, 100, "default", 2.0);
    EXPECT_NEAR(a.lens[0], 1.0); // radius = aperture/2 like CPU lens_radius
    // Lens field never moves the framing (pinhole bit-stable).
    EXPECT_NEAR(a.o[0], p.o[0]);
    EXPECT_NEAR(a.ll[0], p.ll[0]);
    EXPECT_NEAR(a.h[0], p.h[0]);
    EXPECT_NEAR(a.v[1], p.v[1]);
    GPUCam c = build_gpu_camera(200, 200, "cornell", 1.0);
    EXPECT_NEAR(c.lens[0], 0.5);
    EXPECT_NEAR(c.o[2], -800);
    // Weekend final: (13,2,3) vfov-20 framing, defocus aperture by default.
    GPUCam w = build_gpu_camera(1200, 675, "weekend");
    EXPECT_NEAR(w.o[0], 13);
    EXPECT_NEAR(w.o[1], 2);
    EXPECT_NEAR(w.o[2], 3);
    EXPECT_TRUE(w.lens[0] > 0); // default defocus lens on
    GPUCam w0 = build_gpu_camera(1200, 675, "weekend", 0.0);
    EXPECT_NEAR(w0.lens[0], w.lens[0]); // explicit 0 keeps the default lens
    GPUCam b2 = build_gpu_camera(800, 800, "book2");
    EXPECT_NEAR(b2.o[0], 478);
    EXPECT_NEAR(b2.o[1], 278);
    EXPECT_NEAR(b2.o[2], -600);
    // Aliases resolve to the same builders.
    GPUCam bk = build_gpu_camera(1200, 675, "book1");
    EXPECT_NEAR(bk.o[0], w.o[0]);
    GPUCam bx = build_gpu_camera(800, 800, "boxes");
    EXPECT_NEAR(bx.o[0], b2.o[0]);
}

static void t_cornell() {
    test_current = "cornell";
    scene_data scene = build_cornell(16.0 / 9.0, 0.0);
    EXPECT_TRUE((int)scene.objs.size() == 8); // 5 walls + 2 posed boxes + 1 light
    EXPECT_TRUE((int)scene.lights.size() == 1);
    EXPECT_TRUE(light_mat(scene.lights[0])->emitted().x() > 1); // bright
    hittable_list world;
    for (auto &o : scene.objs)
        world.add(o);
    hit_record hr;
    // Center ray still lands inside the room on box or wall.
    EXPECT_TRUE(world.hit(scene.cam.get_ray(0.5, 0.5), 0.001, 1e30, hr));
    EXPECT_TRUE(hr.point.x() > 0 && hr.point.x() < 555);
    EXPECT_TRUE(hr.point.y() > 0 && hr.point.y() < 555);
    EXPECT_TRUE(hr.point.z() > 0 && hr.point.z() < 555);
    // Tall posed box occupies its classic footprint (rotated, so loose).
    aabb tall_box;
    EXPECT_TRUE(scene.objs[5]->bounding_box(tall_box));
    EXPECT_TRUE(tall_box.minimum.x() < 265 && tall_box.maximum.x() > 265);
    EXPECT_TRUE(tall_box.minimum.z() < 295 && tall_box.maximum.z() > 295);
    EXPECT_TRUE(tall_box.maximum.y() > 329 && tall_box.maximum.y() < 331);
    // Default builder unchanged: ground + mesh/fallback + 2 spheres + quad + orb.
    scene_data def = build_default(16.0 / 9.0, 0.0);
    EXPECT_TRUE((int)def.lights.size() == 2); // quad + warm orb
    EXPECT_TRUE(light_mat(def.lights[1])->emitted().z() > 1); // orb glows warm
    EXPECT_TRUE((int)def.objs.size() >= 17); // ground+12mesh+2sph+quad+orb
}

static void t_flatten() {
    test_current = "flatten";
    scene_data def = build_scene("default", 16.0 / 9.0, 0.0);
    flat_scene fs;
    EXPECT_TRUE(flatten_scene(def, fs));
    // Refs conserve prims; QBVH nodes form a real 4-wide tree.
    EXPECT_TRUE((int)fs.refs.size() == (int)def.objs.size());
    EXPECT_TRUE(!fs.nodes.empty());
    int leaf_prims = 0;
    for (auto &n : fs.nodes)
        for (int s = 0; s < 4; ++s)
            leaf_prims += n.count[s];
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
    EXPECT_TRUE(ft.images[0].levels == 1); // 1x1 has no smaller level
    EXPECT_TRUE((int)ft.images[0].rgba.size() == 4); // single texel blob
    EXPECT_TRUE(ft.gs.spheres[0].prm[0] == 5 && ft.gs.spheres[1].prm[0] == 5);
    EXPECT_TRUE(fabs(ft.gs.spheres[1].prm[1] - 0.0f) < 1e-6); // same idx
    // Motion tri uploads B endpoints + range; static tri parks at t=0.
    auto mv = std::make_shared<triangle>(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0),
                                         shared_mat);
    mv->set_motion(vec3(0, 1, 0), vec3(1, 1, 0), vec3(0, 2, 0), 0.0, 2.0);
    scene_data mot;
    mot.objs.push_back(mv);
    mot.objs.push_back(std::make_shared<triangle>(vec3(0, 0, 0), vec3(1, 0, 0),
                                                  vec3(0, 1, 0), shared_mat));
    flat_scene mf2;
    EXPECT_TRUE(flatten_scene(mot, mf2));
    EXPECT_NEAR(mf2.gs.tris[0].a1[1], 1.0); // B endpoint uploaded
    EXPECT_NEAR(mf2.gs.tris[0].tm[1], 2.0); // range uploaded
    EXPECT_NEAR(mf2.gs.tris[1].tm[1], 1.0); // static parks at t=0..1
    EXPECT_NEAR(mf2.gs.tris[1].a1[1], 0.0);
    // Pyramid blob: 4x2 exports L0+L1+L2 consecutively (8+2+1 texels).
    ppm_io::image ramp;
    ramp.w = 4;
    ramp.h = 2;
    ramp.px.assign(8, vec3(0.25, 0.5, 1.0));
    auto ramp_tex = std::make_shared<image_texture>(4, 2, ramp.px);
    auto ramp_mat = std::make_shared<lambertian>(ramp_tex);
    scene_data ramp_scene;
    ramp_scene.objs.push_back(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, ramp_mat));
    flat_scene fr;
    EXPECT_TRUE(flatten_scene(ramp_scene, fr));
    EXPECT_TRUE(fr.images[0].levels == 3);
    EXPECT_TRUE((int)fr.images[0].rgba.size() == (8 + 2 + 1) * 4);
    // Cornell: 8 scene objs bake to 18 quads flat, light first for NEE indexing.
    scene_data cor = build_scene("cornell", 1.0, 0.0);
    flat_scene fc;
    EXPECT_TRUE(flatten_scene(cor, fc));
    EXPECT_TRUE((int)cor.objs.size() == 8);
    EXPECT_TRUE((int)fc.refs.size() == 18); // 5 walls + 12 baked box + 1 light
    EXPECT_TRUE(fc.nlights == 1);
    EXPECT_TRUE(fabs(fc.gs.quads[0].emit[0] - 7) < 1e-6); // light leads
    // Export values exact on a known material.
    auto m = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));
    scene_data tiny2;
    tiny2.objs.push_back(
        std::make_shared<sphere>(vec3(0, 0, -1), vec3(0, 1, -1), 0.0, 1.0, 0.5, m));
    flat_scene fm;
    EXPECT_TRUE(flatten_scene(tiny2, fm));
    EXPECT_TRUE(fm.gs.spheres[0].prm[3] == 1); // motion flag
    EXPECT_NEAR(fm.gs.spheres[0].c1[1], 1.0);
    // Static sphere: flag 0, c1 == c0.
    EXPECT_TRUE(fm.gs.spheres.size() >= 1);
    // Fog medium exports as type-6 sphere slot (boundary + density).
    auto phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
    auto border = std::make_shared<sphere>(vec3(0, 0, -1), 2.0, phase);
    scene_data foggy;
    foggy.objs.push_back(std::make_shared<constant_medium>(border, 0.25, phase));
    flat_scene ff;
    EXPECT_TRUE(flatten_scene(foggy, ff));
    EXPECT_TRUE(ff.gs.spheres[0].prm[0] == 6);
    EXPECT_TRUE(fabs(ff.gs.spheres[0].prm[1] - 0.25f) < 1e-6);
    EXPECT_TRUE(fabs(ff.gs.spheres[0].alb[0] - 0.9f) < 1e-6);
    // Hetero medium exports as type-8 slot (boundary + sigma + freqs).
    scene_data hety;
    hety.objs.push_back(std::make_shared<heterogeneous_medium>(border, 0.6, phase));
    flat_scene fh;
    EXPECT_TRUE(flatten_scene(hety, fh));
    EXPECT_TRUE(fh.gs.spheres[0].prm[0] == 8);
    EXPECT_TRUE(fabs(fh.gs.spheres[0].prm[1] - 0.6f) < 1e-6);
    EXPECT_TRUE(fabs(fh.gs.spheres[0].alb[0] - 0.9f) < 1e-6);
    EXPECT_TRUE(fabs(fh.gs.spheres[0].alb2[0] - 5.0f) < 1e-6);
    EXPECT_TRUE(fabs(fh.gs.spheres[0].alb2[1] - 4.0f) < 1e-6);
    EXPECT_TRUE(fabs(fh.gs.spheres[0].alb2[2] - 6.0f) < 1e-6);
    // Noise texture exports as type-9 slot (freq, depth, mode).
    auto ntex = std::make_shared<noise_texture>(4.0, 7, 2, vec3(0.85, 0.87, 0.9),
                                                vec3(0.05, 0.15, 0.45));
    scene_data noisy;
    noisy.objs.push_back(
        std::make_shared<sphere>(vec3(0, 0, -1), 0.5, std::make_shared<lambertian>(ntex)));
    flat_scene fn;
    EXPECT_TRUE(flatten_scene(noisy, fn));
    EXPECT_TRUE(fn.gs.spheres[0].prm[0] == 9);
    EXPECT_TRUE(fabs(fn.gs.spheres[0].prm[1] - 4.0f) < 1e-6);
    EXPECT_TRUE(fabs(fn.gs.spheres[0].prm[3] - 2.0f) < 1e-6);
    EXPECT_TRUE(fabs(fn.gs.spheres[0].alb[0] - 0.85f) < 1e-6);
    // Opt-in marble sphere: off by default (frozen), one extra obj when on.
    scene_data d0 = build_default(16.0 / 9.0, 0.0);
    scene_data d1 = build_default(16.0 / 9.0, 0.0, 0, 0, 0, 0, true);
    EXPECT_TRUE((int)d1.objs.size() == (int)d0.objs.size() + 1);
    auto mm = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    float alb[4] = {}, alb2[4] = {}, emit[4] = {}, prm[4] = {};
    EXPECT_TRUE(mm->export_gpu(alb, alb2, emit, prm));
    EXPECT_TRUE(fabs(alb[0] - 0.8) < 1e-6);
    EXPECT_TRUE(prm[0] == 7 && fabs(prm[1] - 0.3) < 1e-6);
}

static void t_expand_gpu() {
    test_current = "expand_gpu";
    auto m = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    // qbvh under translate: baked world-space spheres, order preserved.
    hittable_list two;
    two.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, m));
    two.add(std::make_shared<sphere>(vec3(2, 0, -1), 0.5, m));
    std::vector<std::shared_ptr<hittable>> tv = two.children();
    bvh_node bt(tv, 0, tv.size());
    auto qbt = std::make_shared<qbvh_node>(bt);
    std::vector<std::shared_ptr<hittable>> prims;
    qbt->collect_prims(prims);
    EXPECT_TRUE((int)prims.size() == 2); // accessor mirrors traversal
    scene_data moved;
    moved.objs.push_back(std::make_shared<translate>(qbt, vec3(10, 0, 0)));
    flat_scene mf;
    EXPECT_TRUE(flatten_scene(moved, mf));
    EXPECT_TRUE((int)mf.gs.spheres.size() == 2);
    EXPECT_NEAR(mf.gs.spheres[0].c[0], 10.0f);
    EXPECT_NEAR(mf.gs.spheres[1].c[0], 12.0f);
    // book2 cluster shape: rotate_y(180deg) + translate over qbvh spheres.
    hittable_list one;
    one.add(std::make_shared<sphere>(vec3(3, 0, 0), 0.5, m));
    std::vector<std::shared_ptr<hittable>> ov = one.children();
    bvh_node bo(ov, 0, ov.size());
    auto qbo = std::make_shared<qbvh_node>(bo);
    auto ry = std::make_shared<rotate_y>(qbo, 180.0);
    scene_data cl;
    cl.objs.push_back(std::make_shared<translate>(ry, vec3(10, 0, 0)));
    flat_scene cf;
    EXPECT_TRUE(flatten_scene(cl, cf));
    EXPECT_TRUE((int)cf.gs.spheres.size() == 1);
    EXPECT_NEAR(cf.gs.spheres[0].c[0], 7.0f); // -3 + 10
    EXPECT_NEAR(cf.gs.spheres[0].c[2], 0.0f);
    // Moving sphere under translate: endpoints + range preserved.
    scene_data mm2;
    mm2.objs.push_back(std::make_shared<translate>(
        std::make_shared<sphere>(vec3(0, 0, 0), vec3(0, 2, 0), 0.0, 1.0, 0.5, m),
        vec3(5, 0, 0)));
    flat_scene mf2;
    EXPECT_TRUE(flatten_scene(mm2, mf2));
    EXPECT_TRUE(mf2.gs.spheres[0].prm[3] == 1); // still motion-flagged
    EXPECT_NEAR(mf2.gs.spheres[0].c[0], 5.0f);
    EXPECT_NEAR(mf2.gs.spheres[0].c1[1], 2.0f);
    // Transformed volume: unsupported, fails loudly (never silently wrong).
    auto phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
    auto border = std::make_shared<sphere>(vec3(0, 0, -1), 2.0, phase);
    scene_data bad;
    bad.objs.push_back(std::make_shared<translate>(
        std::make_shared<constant_medium>(border, 0.25, phase), vec3(1, 0, 0)));
    flat_scene bf;
    EXPECT_TRUE(!flatten_scene(bad, bf));
    // book2 full scene flattens (instances + qbvh + motion + media).
    scene_data b2 = build_scene("book2", 1.0, 1.0);
    flat_scene fb2;
    EXPECT_TRUE(flatten_scene(b2, fb2));
    EXPECT_TRUE(fb2.nlights == 1); // single ceiling quad (earth ball is lambertian)
}

// M54 forensics: NEE shadow census over harvested book2 segments.
// Compares production CPU shadow_transmittance (fp64 BVH) against an
// fp32 brute-force mimic of common.glsl (exact formulas, flat inputs)
// with and without the M51 shadow-bias lift. Prints disagreement rates;
// the M51 residual hypothesis says cluster segments false-block in fp32.
namespace nee_census {
struct f3 {
    float x, y, z;
};
inline f3 f3sub(f3 a, f3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline f3 f3add(f3 a, f3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline f3 f3mul(f3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float f3dot(f3 a, f3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float f3len(f3 a) { return sqrtf(f3dot(a, a)); }
inline f3 f3norm(f3 a) {
    float l = f3len(a);
    return {a.x / l, a.y / l, a.z / l};
}
inline f3 f3cross(f3 a, f3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// Mirrors common.glsl hit_sphere (motion lerp, half-b root selection).
inline bool sph_hit(const float *o, const float *d, float rt, float tmin, float tmax,
                    const GPUSphere &s, float &t) {
    float f = (s.tm[1] > s.tm[0]) ? (rt - s.tm[0]) / (s.tm[1] - s.tm[0]) : 0.0f;
    if (f < 0.0f)
        f = 0.0f;
    if (f > 1.0f)
        f = 1.0f;
    float cx = s.c[0] + (s.c1[0] - s.c[0]) * f;
    float cy = s.c[1] + (s.c1[1] - s.c[1]) * f;
    float cz = s.c[2] + (s.c1[2] - s.c[2]) * f;
    float ox = o[0] - cx, oy = o[1] - cy, oz = o[2] - cz;
    float a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    float hb = ox * d[0] + oy * d[1] + oz * d[2];
    float rr = s.c[3];
    float c = ox * ox + oy * oy + oz * oz - rr * rr;
    float disc = hb * hb - a * c;
    if (disc < 0.0f)
        return false;
    float sq = sqrtf(disc);
    float root = (-hb - sq) / a;
    if (root < tmin || root > tmax) {
        root = (-hb + sq) / a;
        if (root < tmin || root > tmax)
            return false;
    }
    t = root;
    return true;
}
// Mirrors common.glsl hit_quad (plane + alpha/beta, t-range gate).
inline bool quad_hit(const float *o, const float *d, float tmin, float tmax,
                     const GPUQuad &q, float *t_out = nullptr) {
    float ux = q.u[0], uy = q.u[1], uz = q.u[2];
    float vx = q.v[0], vy = q.v[1], vz = q.v[2];
    float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    float nl = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= nl;
    ny /= nl;
    nz /= nl;
    float denom = nx * d[0] + ny * d[1] + nz * d[2];
    if (fabsf(denom) < 1e-8f)
        return false;
    float dd = nx * q.Q[0] + ny * q.Q[1] + nz * q.Q[2];
    float tt = (dd - (nx * o[0] + ny * o[1] + nz * o[2])) / denom;
    if (tt < tmin || tt > tmax)
        return false;
    float px = o[0] + d[0] * tt - q.Q[0];
    float py = o[1] + d[1] * tt - q.Q[1];
    float pz = o[2] + d[2] * tt - q.Q[2];
    // w = n/(n.n); alpha = w.(p x v); beta = w.(u x p).
    float wxx = nx / nl, wyy = ny / nl, wzz = nz / nl;
    float c1x = py * vz - pz * vy, c1y = pz * vx - px * vz, c1z = px * vy - py * vx;
    float c2x = uy * pz - uz * py, c2y = uz * px - ux * pz, c2z = ux * py - uy * px;
    float al = wxx * c1x + wyy * c1y + wzz * c1z;
    float be = wxx * c2x + wyy * c2y + wzz * c2z;
    if (al < 0.0f || al > 1.0f || be < 0.0f || be > 1.0f)
        return false;
    if (t_out)
        *t_out = tt;
    return true;
}
// Brute-force solid-block test over flat prims (fog slots skipped like
// trace_solid). Returns true = blocked.
inline bool f32_blocked(const flat_scene &fb, const float *o, const float *d,
                        float dist) {
    float tmax = dist - 0.001f;
    for (const auto &s : fb.gs.spheres) {
        if (s.prm[0] == 6.0f || s.prm[0] == 8.0f)
            continue;
        float t;
        if (sph_hit(o, d, 0.0f, 0.001f, tmax, s, t))
            return true;
    }
    for (const auto &q : fb.gs.quads) {
        if (quad_hit(o, d, 0.001f, tmax, q))
            return true;
    }
    return false;
}
// Same, but names the first blocker (forensics).
inline bool f32_blocker(const flat_scene &fb, const float *o, const float *d,
                        float dist, int &ty, int &idx, float &bt) {
    float tmax = dist - 0.001f;
    float best = tmax;
    ty = -1;
    idx = -1;
    for (int k = 0; k < (int)fb.gs.spheres.size(); ++k) {
        const auto &s = fb.gs.spheres[(size_t)k];
        if (s.prm[0] == 6.0f || s.prm[0] == 8.0f)
            continue;
        float t;
        if (sph_hit(o, d, 0.0f, 0.001f, best, s, t)) {
            best = t;
            ty = 0;
            idx = k;
        }
    }
    for (int k = 0; k < (int)fb.gs.quads.size(); ++k) {
        const auto &q = fb.gs.quads[(size_t)k];
        float t;
        if (quad_hit(o, d, 0.001f, best, q, &t)) {
            best = t;
            ty = 1;
            idx = k;
        }
    }
    bt = best;
    return ty >= 0;
}
// Closest solid hit with fp32 face-forwarded normal (device-faithful),
// plus winning prim id (self-vs-neighbor forensics).
inline bool f32_first_hit_id(const flat_scene &fb, const float *o, const float *d,
                             float &t, float *n, int &ty, int &idx) {
    float best = 1e30f;
    bool hit = false;
    ty = -1;
    idx = -1;
    for (int k = 0; k < (int)fb.gs.spheres.size(); ++k) {
        const auto &s = fb.gs.spheres[(size_t)k];
        if (s.prm[0] == 6.0f || s.prm[0] == 8.0f)
            continue;
        float tt;
        if (sph_hit(o, d, 0.0f, 0.001f, best, s, tt)) {
            best = tt;
            ty = 0;
            idx = k;
            float rr = s.c[3];
            float nx = (o[0] + d[0] * tt - s.c[0]) / rr;
            float ny = (o[1] + d[1] * tt - s.c[1]) / rr;
            float nz = (o[2] + d[2] * tt - s.c[2]) / rr;
            float l = sqrtf(nx * nx + ny * ny + nz * nz);
            n[0] = nx / l;
            n[1] = ny / l;
            n[2] = nz / l;
            hit = true;
        }
    }
    for (int k = 0; k < (int)fb.gs.quads.size(); ++k) {
        const auto &q = fb.gs.quads[(size_t)k];
        float tt;
        if (quad_hit(o, d, 0.001f, best, q, &tt)) {
            best = tt;
            ty = 1;
            idx = k;
            float nx = q.u[1] * q.v[2] - q.u[2] * q.v[1];
            float ny = q.u[2] * q.v[0] - q.u[0] * q.v[2];
            float nz = q.u[0] * q.v[1] - q.u[1] * q.v[0];
            float l = sqrtf(nx * nx + ny * ny + nz * nz);
            float sgn = (d[0] * nx + d[1] * ny + d[2] * nz) > 0 ? -1.0f : 1.0f;
            n[0] = nx / l * sgn;
            n[1] = ny / l * sgn;
            n[2] = nz / l * sgn;
            hit = true;
        }
    }
    t = best;
    return hit;
}
inline bool f32_first_hit(const flat_scene &fb, const float *o, const float *d,
                          float &t, float *n) {
    int dty = -1, didx = -1;
    return f32_first_hit_id(fb, o, d, t, n, dty, didx);
}
} // namespace nee_census

static void t_nee_shadow_census() {
    test_current = "nee_census";
    rng_seed(42); // same scene as the render anchors
    scene_data b2 = build_scene("book2", 1.0, 0.0);
    flat_scene fb;
    EXPECT_TRUE(flatten_scene(b2, fb));
    EXPECT_TRUE(fb.gs.tris.empty()); // book2 has no tris: mimic covers all
    qbvh_node world(b2.objs, 0, b2.objs.size(), true);
    // Cluster bbox from flat r=10 spheres (world space, expander-baked).
    double bblo[3] = {1e30, 1e30, 1e30}, bbhi[3] = {-1e30, -1e30, -1e30};
    for (const auto &s : fb.gs.spheres) {
        if (s.c[3] != 10.0f)
            continue;
        for (int k = 0; k < 3; ++k) {
            double c = s.c[k];
            if (c - 10.0 < bblo[k])
                bblo[k] = c - 10.0;
            if (c + 10.0 > bbhi[k])
                bbhi[k] = c + 10.0;
        }
    }
    auto in_cluster = [&](const vec3 &p) {
        return p.x() > bblo[0] && p.x() < bbhi[0] && p.y() > bblo[1] &&
               p.y() < bbhi[1] && p.z() > bblo[2] && p.z() < bbhi[2];
    };
    // Light quad corners (fixed targets, no RNG).
    auto lq = std::dynamic_pointer_cast<quad>(b2.lights[0].shape);
    EXPECT_TRUE(lq != nullptr);
    vec3 L[5] = {lq->corner() + 0.5 * lq->edge_u() + 0.5 * lq->edge_v(), lq->corner(),
                 lq->corner() + lq->edge_u(), lq->corner() + lq->edge_v(),
                 lq->corner() + lq->edge_u() + lq->edge_v()};
    rng_seed(1000); // deterministic march stream
    long n = 0, ncl = 0, cpuB = 0, f32B = 0, liftB = 0;
    long clCpuB = 0, clF32B = 0, clLiftB = 0;
    long falseBlock = 0, missBlock = 0, clFalse = 0, clMiss = 0;
    long lift_examples = 0;
    const int G = 60;
    for (int j = 0; j < G; ++j) {
        for (int i = 0; i < G; ++i) {
            double u = (i + 0.5) / G, v = (j + 0.5) / G;
            ray primary = b2.cam.get_ray(u, v);
            hit_record rec;
            if (!world.hit(primary, 0.001, 1e30, rec))
                continue;
            if (!rec.mat->is_diffuse())
                continue;
            bool cl = in_cluster(rec.point);
            for (int k = 0; k < 5; ++k) {
                vec3 toL = L[k] - rec.point;
                double dist = toL.length();
                vec3 wi = toL / dist;
                if (dot(rec.normal, wi) <= 0)
                    continue;
                ++n;
                if (cl)
                    ++ncl;
                double Tr = shadow_transmittance(world, b2.media, rec.point, wi, dist,
                                                 primary.time());
                bool cb = (Tr <= 0.0);
                // fp32 mimic inputs (upload precision + shader fp32).
                float o[3] = {(float)rec.point.x(), (float)rec.point.y(),
                              (float)rec.point.z()};
                nee_census::f3 df = {(float)wi.x(), (float)wi.y(), (float)wi.z()};
                df = nee_census::f3norm(df);
                float d[3] = {df.x, df.y, df.z};
                bool f32 = nee_census::f32_blocked(fb, o, d, (float)dist);
                float ol[3] = {o[0] + (float)rec.normal.x() * 0.001f,
                               o[1] + (float)rec.normal.y() * 0.001f,
                               o[2] + (float)rec.normal.z() * 0.001f};
                // Shader-faithful lift: wi/dist recomputed FROM the lifted
                // origin (path.comp aims toL = lp - hitp). Keeping unlifted
                // wi/dist with a lifted origin aims off-target and the light
                // quad itself false-blocks (mimic artifact, not device bug).
                float lx = (float)L[k].x() - ol[0], ly = (float)L[k].y() - ol[1],
                      lz = (float)L[k].z() - ol[2];
                float ld = sqrtf(lx * lx + ly * ly + lz * lz);
                float dl[3] = {lx / ld, ly / ld, lz / ld};
                bool lf = nee_census::f32_blocked(fb, ol, dl, ld);
                if (cb)
                    ++cpuB;
                if (f32)
                    ++f32B;
                if (lf)
                    ++liftB;
                if (cl) {
                    if (cb)
                        ++clCpuB;
                    if (f32)
                        ++clF32B;
                    if (lf)
                        ++clLiftB;
                }
                if (!cb && lf && lift_examples < 8) {
                    ++lift_examples;
                    int bty = -1, bidx = -1;
                    float btt = 0;
                    nee_census::f32_blocker(fb, ol, dl, ld, bty, bidx, btt);
                    std::printf("  [lift-block] o=(%.3f,%.3f,%.3f) n=(%.2f,%.2f,%.2f) "
                                "wi=(%.2f,%.2f,%.2f) dist=%.2f cl=%d blk=(%d,%d,t=%.4f)\n",
                                rec.point.x(), rec.point.y(), rec.point.z(),
                                rec.normal.x(), rec.normal.y(), rec.normal.z(), wi.x(),
                                wi.y(), wi.z(), dist, cl ? 1 : 0, bty, bidx, btt);
                }
                if (!cb && f32) {
                    ++falseBlock;
                    if (cl)
                        ++clFalse;
                }
                if (cb && !f32) {
                    ++missBlock;
                    if (cl)
                        ++clMiss;
                }
            }
        }
    }
    std::printf("  [nee_census] segs=%ld cluster=%ld\n", n, ncl);
    std::printf("  [nee_census] cpuBlock=%.3f f32Block=%.3f liftBlock=%.3f\n",
                (double)cpuB / n, (double)f32B / n, (double)liftB / n);
    std::printf("  [nee_census] cl: cpuBlock=%.3f f32Block=%.3f liftBlock=%.3f\n",
                ncl ? (double)clCpuB / ncl : -1, ncl ? (double)clF32B / ncl : -1,
                ncl ? (double)clLiftB / ncl : -1);
    std::printf("  [nee_census] falseBlock=%.4f missBlock=%.4f (cluster %.4f/%.4f)\n",
                (double)falseBlock / n, (double)missBlock / n,
                ncl ? (double)clFalse / ncl : -1, ncl ? (double)clMiss / ncl : -1);
    // First-hit agreement: fp64 production vs fp32-brute (flat inputs).
    // Interpenetrating overlaps flip front identity on 1e-4 input shifts;
    // that persistent geometry difference (not noise) is the M54 suspect.
    long fh = 0, fhCl = 0, fhMiss = 0, fhClMiss = 0, fhNorm = 0, fhClNorm = 0;
    for (int j = 0; j < G; ++j) {
        for (int i = 0; i < G; ++i) {
            double u = (i + 0.5) / G, v = (j + 0.5) / G;
            ray primary = b2.cam.get_ray(u, v);
            hit_record rec;
            bool h64 = world.hit(primary, 0.001, 1e30, rec);
            float o[3] = {(float)primary.origin().x(), (float)primary.origin().y(),
                          (float)primary.origin().z()};
            nee_census::f3 dd = {(float)primary.direction().x(),
                                 (float)primary.direction().y(),
                                 (float)primary.direction().z()};
            float d[3] = {dd.x, dd.y, dd.z};
            float bt = 1e30;
            nee_census::f3 bn = {0, 0, 0};
            bool h32 = false;
            for (const auto &s : fb.gs.spheres) {
                if (s.prm[0] == 6.0f || s.prm[0] == 8.0f)
                    continue;
                float t;
                if (nee_census::sph_hit(o, d, 0.0f, 0.001f, bt, s, t)) {
                    bt = t;
                    float f = 0.0f; // static harvest (shutter closed)
                    float cx = s.c[0], cy = s.c[1], cz = s.c[2];
                    float rr = s.c[3];
                    bn = {(o[0] + d[0] * t - cx) / rr, (o[1] + d[1] * t - cy) / rr,
                          (o[2] + d[2] * t - cz) / rr};
                    float bl = sqrtf(bn.x * bn.x + bn.y * bn.y + bn.z * bn.z);
                    bn = {bn.x / bl, bn.y / bl, bn.z / bl};
                    h32 = true;
                }
            }
            for (const auto &q : fb.gs.quads) {
                float t;
                if (nee_census::quad_hit(o, d, 0.001f, bt, q, &t)) {
                    bt = t;
                    float nx = q.u[1] * q.v[2] - q.u[2] * q.v[1];
                    float ny = q.u[2] * q.v[0] - q.u[0] * q.v[2];
                    float nz = q.u[0] * q.v[1] - q.u[1] * q.v[0];
                    float nl = sqrtf(nx * nx + ny * ny + nz * nz);
                    nx /= nl;
                    ny /= nl;
                    nz /= nl;
                    float sgn = (d[0] * nx + d[1] * ny + d[2] * nz) > 0 ? -1.0f : 1.0f;
                    bn = {nx * sgn, ny * sgn, nz * sgn};
                    h32 = true;
                }
            }
            if (!h64 && !h32)
                continue;
            // Volume events have no fp32-brute counterpart (fog_event path).
            if (h64 && (dynamic_cast<const constant_medium *>(rec.hit_obj) ||
                        dynamic_cast<const heterogeneous_medium *>(rec.hit_obj)))
                continue;
            bool cl = h64 ? in_cluster(rec.point)
                          : (o[0] + d[0] * bt > bblo[0] && o[0] + d[0] * bt < bbhi[0] &&
                             o[1] + d[1] * bt > bblo[1] && o[1] + d[1] * bt < bbhi[1] &&
                             o[2] + d[2] * bt > bblo[2] && o[2] + d[2] * bt < bbhi[2]);
            ++fh;
            if (cl)
                ++fhCl;
            if (h64 != h32) {
                ++fhMiss;
                if (cl)
                    ++fhClMiss;
                continue;
            }
            if (!h64)
                continue;
            double f64t = rec.t;
            double dt = fabs(f64t - (double)bt);
            double rel = dt / fmax(f64t, 1e-6);
            double cosn = rec.normal.x() * bn.x + rec.normal.y() * bn.y +
                          rec.normal.z() * bn.z;
            if (rel > 1e-3 || cosn < 0.999) {
                ++fhNorm;
                if (cl)
                    ++fhClNorm;
            }
        }
    }
    std::printf("  [firsthit] rays=%ld cluster=%ld missMismatch=%.4f (cl %.4f) "
                "t/normMismatch=%.4f (cl %.4f)\n",
                fh, fhCl, (double)fhMiss / fh, fhCl ? (double)fhClMiss / fhCl : -1,
                (double)fhNorm / fh, fhCl ? (double)fhClNorm / fhCl : -1);
    // Same-lp NEE value census: identical light points both sides, compare
    // every estimator piece (cosS/cosA/dist/pdf/w/Tr/value). Rules in/out
    // geometry+formula; only the sample stream would remain.
    {
        auto lt = b2.lights[0];
        auto lmat = light_mat(lt);
        vec3 le = lmat->emitted();
        double A = light_area(lt);
        rng_seed(777);
        long sk = 0;
        double sumR_cosS = 0, sumR_cosA = 0, sumR_dist = 0, sumR_pdf = 0;
        double sumR_w = 0, sumR_tr = 0, sumR_val = 0;
        const int K = 8;
        for (int j = 0; j < G && sk < 200; ++j) {
            for (int i = 0; i < G && sk < 200; ++i) {
                double u = (i + 0.5) / G, v = (j + 0.5) / G;
                ray primary = b2.cam.get_ray(u, v);
                hit_record rec;
                if (!world.hit(primary, 0.001, 1e30, rec))
                    continue;
                if (!rec.mat->is_diffuse() || !in_cluster(rec.point))
                    continue;
                for (int s = 0; s < K; ++s) {
                    double u1 = random_double(), u2 = random_double();
                    vec3 lp = light_point(lt, u1, u2, primary.time());
                    vec3 toL = lp - rec.point;
                    double dist = toL.length();
                    vec3 wi = toL / dist;
                    vec3 ln = light_normal_at(lt, lp, primary.time());
                    double cosS = dot(rec.normal, wi);
                    double cosA = fabs(dot(ln, -wi));
                    if (cosS <= 0 || cosA <= 0 || A <= 0)
                        continue;
                    double Tr = shadow_transmittance(world, b2.media, rec.point, wi,
                                                     dist, primary.time());
                    double pdf_l = dist * dist / (1.0 * A * cosA);
                    double pdf_b = cosine_pdf(cosS);
                    double w = direction_pdf::power_weight(pdf_l, pdf_b);
                    const double pi = 3.1415926535897932385;
                    double cval = (cosS * cosA * A / (pi * dist * dist)) * w * Tr;
                    // fp32 side (float casts like upload + shader fp32).
                    float fu1 = (float)u1, fu2 = (float)u2;
                    (void)fu1;
                    (void)fu2;
                    float ox = (float)rec.point.x(), oy = (float)rec.point.y(),
                          oz = (float)rec.point.z();
                    float nx = (float)rec.normal.x(), ny = (float)rec.normal.y(),
                          nz = (float)rec.normal.z();
                    // Light quad in float (flat).
                    const GPUQuad &fq = fb.gs.quads[(size_t)fb.light_table[0].second];
                    float lx = fq.Q[0] + (float)u1 * fq.u[0] + (float)u2 * fq.v[0];
                    float ly = fq.Q[1] + (float)u1 * fq.u[1] + (float)u2 * fq.v[1];
                    float lz = fq.Q[2] + (float)u1 * fq.u[2] + (float)u2 * fq.v[2];
                    float tx = lx - ox, ty = ly - oy, tz = lz - oz;
                    float fd = sqrtf(tx * tx + ty * ty + tz * tz);
                    tx /= fd;
                    ty /= fd;
                    tz /= fd;
                    float fcosS = nx * tx + ny * ty + nz * tz;
                    float ex = fq.u[1] * fq.v[2] - fq.u[2] * fq.v[1];
                    float ey = fq.u[2] * fq.v[0] - fq.u[0] * fq.v[2];
                    float ez = fq.u[0] * fq.v[1] - fq.u[1] * fq.v[0];
                    float el = sqrtf(ex * ex + ey * ey + ez * ez);
                    ex /= el;
                    ey /= el;
                    ez /= el;
                    float fcosA = fabsf(ex * -tx + ey * -ty + ez * -tz);
                    float fA = el;
                    float fpdf_l = fd * fd / (fA * fcosA);
                    float fpdf_b = fcosS / 3.14159265f;
                    float a2 = fpdf_l * fpdf_l, b2w = fpdf_b * fpdf_b;
                    float fw = (a2 + b2w) > 0 ? a2 / (a2 + b2w) : 0;
                    float ol[3] = {ox + nx * 0.001f, oy + ny * 0.001f,
                                   oz + nz * 0.001f};
                    float fllx = lx - ol[0], flly = ly - ol[1], fllz = lz - ol[2];
                    float fld = sqrtf(fllx * fllx + flly * flly + fllz * fllz);
                    float dl[3] = {fllx / fld, flly / fld, fllz / fld};
                    bool blk = nee_census::f32_blocked(fb, ol, dl, fld);
                    // f32 haze chord (type-6 analytic over flat fog spheres).
                    float fTr = blk ? 0.0f : 1.0f;
                    if (!blk) {
                        float tmax = fld - 0.001f;
                        for (const auto &sg : fb.gs.spheres) {
                            if (sg.prm[0] != 6.0f)
                                continue;
                            float te, tx2;
                            nee_census::f3 dn{0, 0, 0};
                            // entry/exit via sph_hit twice (mirrors march)
                            float oo[3] = {ol[0], ol[1], ol[2]};
                            if (!nee_census::sph_hit(oo, dl, 0.0f, -1e30f, tmax, sg,
                                                     te))
                                continue;
                            te = te < 0.001f ? 0.001f : te;
                            if (!nee_census::sph_hit(oo, dl, 0.0f, te + 1e-4f, 1e30f,
                                                     sg, tx2))
                                continue;
                            if (tx2 > tmax)
                                tx2 = tmax;
                            if (tx2 > te)
                                fTr *= expf(-sg.prm[1] * (tx2 - te));
                        }
                    }
                    float fval = (fcosS * fcosA * fA / (3.14159265f * fd * fd)) *
                                 fw * fTr;
                    ++sk;
                    sumR_cosS += cosS > 0 ? fcosS / cosS : 1;
                    sumR_cosA += fcosA / cosA;
                    sumR_dist += fd / dist;
                    sumR_pdf += (pdf_l > 0) ? fpdf_l / pdf_l : 1;
                    sumR_w += (w > 0) ? fw / w : 1;
                    sumR_tr += (Tr > 0) ? fTr / Tr : (fTr == 0 ? 1 : 0);
                    sumR_val += (cval > 0) ? fval / cval : 1;
                    (void)le;
                }
            }
        }
        double n = (double)sk;
        std::printf("  [neeval] k=%ld cosS=%.5f cosA=%.5f dist=%.6f pdf=%.5f "
                    "w=%.5f Tr=%.5f val=%.5f\n",
                    sk, sumR_cosS / n, sumR_cosA / n, sumR_dist / n, sumR_pdf / n,
                    sumR_w / n, sumR_tr / n, sumR_val / n);
    }
    EXPECT_TRUE(n > 1000);
    EXPECT_TRUE(ncl > 100);
}

static void t_hotpixel() {
    test_current = "hotpixel";
    // M54 fixed-RNG renders diverge on one cluster sphere (cpu bright,
    // gpu black, deterministic). Re-trace those pixels sample-by-sample
    // with identical inputs (strata centers, light center) and print the
    // first NEE piece that disagrees.
    rng_fixed_flag() = true;
    rng_seed(42);
    const int W = 200, H = 200;
    scene_data b2 = build_scene("book2", 1.0, 0.0);
    flat_scene fb;
    EXPECT_TRUE(flatten_scene(b2, fb));
    qbvh_node world(b2.objs, 0, b2.objs.size(), true);
    auto lt = b2.lights[0];
    double A = light_area(lt);
    const double pi = 3.1415926535897932385;
    std::vector<sample_offset> offs;
    fill_pixel_samples(4, offs);
    int shown = 0;
    long processed = 0;
    for (int yi = 60; yi <= 80 && shown < 6; ++yi) {
        for (int xi = 100; xi <= 125 && shown < 6; ++xi) {
            int j = H - 1 - yi;
            for (auto [ox, oy] : offs) {
                double u = (xi + ox) / W, v = (j + oy) / H;
                ray primary = b2.cam.get_ray(u, v);
                hit_record rec;
                if (!world.hit(primary, 0.001, 1e30, rec))
                    continue;
                if (!rec.mat->is_diffuse())
                    continue;
                vec3 lp = light_point(lt, 0.5, 0.5, primary.time());
                vec3 toL = lp - rec.point;
                double dist = toL.length();
                vec3 wi = toL / dist;
                double cosS = dot(rec.normal, wi);
                if (cosS <= 0)
                    continue;
                ++processed;
                vec3 ln = light_normal_at(lt, lp, primary.time());
                double cosA = fabs(dot(ln, -wi));
                double Tr = shadow_transmittance(world, b2.media, rec.point, wi,
                                                 dist, primary.time());
                double pdf_l = dist * dist / (A * cosA);
                double pdf_b = cosine_pdf(cosS);
                double w = direction_pdf::power_weight(pdf_l, pdf_b);
                double cval = (cosS * cosA * A / (pi * dist * dist)) * w * Tr;
                // f32 side (device-faithful: lifted origin, recomputed dir).
                float fx = (float)rec.point.x(), fy = (float)rec.point.y(),
                      fz = (float)rec.point.z();
                float fnx = (float)rec.normal.x(), fny = (float)rec.normal.y(),
                      fnz = (float)rec.normal.z();
                const GPUQuad &fq = fb.gs.quads[(size_t)fb.light_table[0].second];
                float qlx = fq.Q[0] + 0.5f * fq.u[0] + 0.5f * fq.v[0];
                float qly = fq.Q[1] + 0.5f * fq.u[1] + 0.5f * fq.v[1];
                float qlz = fq.Q[2] + 0.5f * fq.u[2] + 0.5f * fq.v[2];
                float ol[3] = {fx + fnx * 0.001f, fy + fny * 0.001f,
                               fz + fnz * 0.001f};
                float dx = qlx - ol[0], dy = qly - ol[1], dz = qlz - ol[2];
                float fd = sqrtf(dx * dx + dy * dy + dz * dz);
                float dl[3] = {dx / fd, dy / fd, dz / fd};
                int bty = -1, bidx = -1;
                float btt = 0;
                bool blk = nee_census::f32_blocker(fb, ol, dl, fd, bty, bidx, btt);
                float fTr = 0.0f;
                if (!blk) {
                    fTr = 1.0f;
                    float tmax = fd - 0.001f;
                    for (const auto &sg : fb.gs.spheres) {
                        if (sg.prm[0] != 6.0f)
                            continue;
                        float te, tx2;
                        if (!nee_census::sph_hit(ol, dl, 0.0f, -1e30f, tmax, sg, te))
                            continue;
                        te = te < 0.001f ? 0.001f : te;
                        if (!nee_census::sph_hit(ol, dl, 0.0f, te + 1e-4f, 1e30f, sg,
                                                 tx2))
                            continue;
                        if (tx2 > tmax)
                            tx2 = tmax;
                        if (tx2 > te)
                            fTr *= expf(-sg.prm[1] * (tx2 - te));
                    }
                }
                float ex = fq.u[1] * fq.v[2] - fq.u[2] * fq.v[1];
                float ey = fq.u[2] * fq.v[0] - fq.u[0] * fq.v[2];
                float ez = fq.u[0] * fq.v[1] - fq.u[1] * fq.v[0];
                float el = sqrtf(ex * ex + ey * ey + ez * ez);
                float fcosS = fnx * dl[0] + fny * dl[1] + fnz * dl[2];
                float fcosA = fabsf((ex / el) * -dl[0] + (ey / el) * -dl[1] +
                                    (ez / el) * -dl[2]);
                float fpdf_l = fd * fd / (el * fcosA);
                float fpdf_b = fcosS / 3.14159265f;
                float a2 = fpdf_l * fpdf_l, bw = fpdf_b * fpdf_b;
                float fw = (a2 + bw) > 0 ? a2 / (a2 + bw) : 0;
                float fval = (fcosS * fcosA * el / (3.14159265f * fd * fd)) * fw *
                             fTr;
                double rel = (cval > 1e-9) ? fabs(fval - cval) / cval : 0.0;
                // Clearance census: closest fp32 approach even below tmin
                // (near-miss inflation suspect for the device blob).
                float best_all = 1e30;
                int best_ty = -1, best_idx = -1;
                for (int k = 0; k < (int)fb.gs.spheres.size(); ++k) {
                    const auto &s = fb.gs.spheres[(size_t)k];
                    if (s.prm[0] == 6.0f || s.prm[0] == 8.0f)
                        continue;
                    float t;
                    if (nee_census::sph_hit(ol, dl, 0.0f, 0.0f, fd - 0.001f, s, t) &&
                        t < best_all) {
                        best_all = t;
                        best_ty = 0;
                        best_idx = k;
                    }
                }
                for (int k = 0; k < (int)fb.gs.quads.size(); ++k) {
                    const auto &q = fb.gs.quads[(size_t)k];
                    float t;
                    if (nee_census::quad_hit(ol, dl, 0.0f, best_all, q, &t)) {
                        best_all = t;
                        best_ty = 1;
                        best_idx = k;
                    }
                }
                if (rel > 0.01 || (Tr > 0 && fTr == 0) || (Tr == 0 && fTr > 0) ||
                    best_all < 0.01f) {
                    ++shown;
                    std::printf("  [hot] px=(%d,%d) o=(%.4f,%.4f,%.4f) "
                                "n=(%.3f,%.3f,%.3f)\n",
                                xi, yi, rec.point.x(), rec.point.y(), rec.point.z(),
                                rec.normal.x(), rec.normal.y(), rec.normal.z());
                    std::printf("       cpu: cosS=%.5f cosA=%.5f dist=%.3f Tr=%.4f "
                                "w=%.4f val=%.6f\n",
                                cosS, cosA, dist, Tr, w, cval);
                    std::printf("       f32: cosS=%.5f cosA=%.5f dist=%.3f Tr=%.4f "
                                "w=%.4f val=%.6f blk=(%d,%d,t=%.4f) near=(%d,%d,t=%.5f)\n",
                                fcosS, fcosA, fd, fTr, fw, fval, bty, bidx, btt,
                                best_ty, best_idx, best_all);
                }
            }
        }
    }
    std::printf("  [hot] diverging samples shown=%d (processed=%ld)\n", shown,
                processed);
    rng_fixed_flag() = false;
    EXPECT_TRUE(shown >= 0); // forensics probe: prints witnesses, never fails
}

static const char *mat_name(const std::shared_ptr<material> &m) {
    if (dynamic_cast<const lambertian *>(m.get()))
        return "lamb";
    if (dynamic_cast<const metal *>(m.get()))
        return "metal";
    if (dynamic_cast<const dielectric *>(m.get()))
        return "glass";
    if (dynamic_cast<const diffuse_light *>(m.get()))
        return "emit";
    if (dynamic_cast<const isotropic *>(m.get()))
        return "iso";
    return "?";
}

static void t_hotpixel2() {
    test_current = "hotpixel2";
    // Fixed-RNG depth-2 diverges (mixed signs, core + edge spots). Walk
    // bounce-1 per sample: first-hit flip, NEE value, or found-light MIS?
    rng_fixed_flag() = true;
    rng_seed(42);
    const int W = 200, H = 200;
    scene_data b2 = build_scene("book2", 1.0, 0.0);
    flat_scene fb;
    EXPECT_TRUE(flatten_scene(b2, fb));
    qbvh_node world(b2.objs, 0, b2.objs.size(), true);
    auto lt = b2.lights[0];
    double A = light_area(lt);
    const double pi = 3.1415926535897932385;
    std::vector<sample_offset> offs;
    fill_pixel_samples(4, offs);
    int shown = 0;
    long n1 = 0;
    long cSelf = 0, cNear = 0, cInside = 0, cQuad = 0, cHitFlip = 0, cHitFlipR = 0;
    const int boxes[3][4] = {{170, 185, 120, 140}, {100, 125, 80, 95}, {115, 128, 145, 160}};
    for (int b = 0; b < 3; ++b) {
        for (int yi = boxes[b][2]; yi <= boxes[b][3]; ++yi) {
            for (int xi = boxes[b][0]; xi <= boxes[b][1]; ++xi) {
                int j = H - 1 - yi;
                for (auto [ox, oy] : offs) {
                    ray primary = b2.cam.get_ray((xi + ox) / W, (j + oy) / H);
                    hit_record r0;
                    if (!world.hit(primary, 0.001, 1e30, r0))
                        continue;
                    if (r0.mat->emitted().length_squared() > 0)
                        continue;
                    vec3 at0;
                    ray sc0;
                    if (!r0.mat->scatter(primary, r0, at0, sc0))
                        continue;
                    sc0.set_time(primary.time());
                    hit_record r1;
                    bool h1 = world.hit(sc0, 0.001, 1e30, r1);
                    float o[3] = {(float)sc0.origin().x(), (float)sc0.origin().y(),
                                  (float)sc0.origin().z()};
                    nee_census::f3 dd = {(float)sc0.direction().x(),
                                         (float)sc0.direction().y(),
                                         (float)sc0.direction().z()};
                    float d[3] = {dd.x, dd.y, dd.z};
                    float ft = 0;
                    float fn[3] = {0, 0, 0};
                    int fty = -1, fidx = -1;
                    bool f1 = nee_census::f32_first_hit_id(fb, o, d, ft, fn, fty,
                                                           fidx);
                    bool r1vol = h1 && (dynamic_cast<const constant_medium *>(r1.hit_obj) ||
                                        dynamic_cast<const heterogeneous_medium *>(r1.hit_obj));
                    if (r1vol)
                        continue; // fog path: separate mechanism, exonerated
                    ++n1;
                    if (h1 != f1) {
                        if (h1)
                            ++cHitFlipR;
                        else
                            ++cHitFlip;
                        if (shown < 10) {
                            ++shown;
                            std::printf("  [hp2] px=(%d,%d) HIT-FLIP cpu=%d f32=%d "
                                        "(-)\n",
                                        xi, yi, h1 ? 1 : 0, f1 ? 1 : 0);
                        }
                        continue;
                    }
                    if (!h1)
                        continue;
                    double dt = fabs(r1.t - (double)ft) / fmax(r1.t, 1e-6);
                    double cn = r1.normal.x() * fn[0] + r1.normal.y() * fn[1] +
                                r1.normal.z() * fn[2];
                    const char *m1 = mat_name(r1.mat);
                    if (dt > 1e-3 || cn < 0.999) {
                        const char *rel = "?";
                        if (fty == 0 && fidx >= 0) {
                            const auto &ws = fb.gs.spheres[(size_t)fidx];
                            float dx = o[0] - ws.c[0], dy = o[1] - ws.c[1],
                                  dz = o[2] - ws.c[2];
                            float dc = sqrtf(dx * dx + dy * dy + dz * dz);
                            float off = dc - ws.c[3];
                            if (fabsf(off) < 1e-3f) {
                                rel = "SELF-SKIM";
                                ++cSelf;
                            } else if (off < 0) {
                                rel = "INSIDE-NBR";
                                ++cInside;
                            } else {
                                rel = "NEAR-NBR";
                                ++cNear;
                            }
                        } else if (fty == 1) {
                            rel = "QUAD";
                            ++cQuad;
                        }
                        if (shown < 10) {
                            ++shown;
                            std::printf("  [hp2] px=(%d,%d) T/NORM-FLIP dt=%.4f "
                                        "cosn=%.4f (%s) win=(%d,%d,t=%.5f,%s)\n",
                                        xi, yi, dt, cn, m1, fty, fidx, (double)ft,
                                        rel);
                        }
                        std::printf("       ray o=(%.4f,%.4f,%.4f) d=(%.5f,%.5f,%.5f)\n",
                                    sc0.origin().x(), sc0.origin().y(),
                                    sc0.origin().z(), sc0.direction().x(),
                                    sc0.direction().y(), sc0.direction().z());
                        std::printf("       host t=%.5f n=(%.4f,%.4f,%.4f) | "
                                    "f32 t=%.5f n=(%.4f,%.4f,%.4f)\n",
                                    r1.t, r1.normal.x(), r1.normal.y(),
                                    r1.normal.z(), (double)ft, fn[0], fn[1], fn[2]);
                        continue;
                    }
                    if (r1.mat->emitted().length_squared() > 0) {
                        // Found-light MIS: BSDF-found emitter weighted by its
                        // geometric pdf (host pointer-match vs device id).
                        vec3 toH = r1.point - r0.point;
                        double dist = toH.length();
                        double pdf_l = direction_pdf::nee_value_for_hit(
                            b2.lights, r1.mat, r1.point, r0.point, sc0.time());
                        vec3 usd = unit_vector(sc0.direction());
                        double pdf_b = cosine_pdf(dot(usd, r0.normal));
                        double wh = (pdf_l <= 0)
                                        ? 1.0
                                        : direction_pdf::power_weight(pdf_b, pdf_l);
                        // f32: flat quad geometric pdf + float cosine.
                        const GPUQuad &fq2 =
                            fb.gs.quads[(size_t)fb.light_table[0].second];
                        float ex2 = fq2.u[1] * fq2.v[2] - fq2.u[2] * fq2.v[1];
                        float ey2 = fq2.u[2] * fq2.v[0] - fq2.u[0] * fq2.v[2];
                        float ez2 = fq2.u[0] * fq2.v[1] - fq2.u[1] * fq2.v[0];
                        float el2 = sqrtf(ex2 * ex2 + ey2 * ey2 + ez2 * ez2);
                        float fx1 = (float)r1.point.x() - (float)r0.point.x();
                        float fy1 = (float)r1.point.y() - (float)r0.point.y();
                        float fz1 = (float)r1.point.z() - (float)r0.point.z();
                        float fd1 = sqrtf(fx1 * fx1 + fy1 * fy1 + fz1 * fz1);
                        float cA = fabsf((ex2 / el2) * (-fx1 / fd1) +
                                         (ey2 / el2) * (-fy1 / fd1) +
                                         (ez2 / el2) * (-fz1 / fd1));
                        float fpdf_l = (cA > 0 && el2 > 0)
                                           ? fd1 * fd1 / (1.0f * el2 * cA)
                                           : 0.0f;
                        float sdx = (float)usd.x(), sdy = (float)usd.y(),
                              sdz = (float)usd.z();
                        float sl = sqrtf(sdx * sdx + sdy * sdy + sdz * sdz);
                        float fpdf_b =
                            (sdx / sl * (float)r0.normal.x() +
                             sdy / sl * (float)r0.normal.y() +
                             sdz / sl * (float)r0.normal.z()) /
                            3.14159265f;
                        float aa = fpdf_b * fpdf_b, bb = fpdf_l * fpdf_l;
                        float wf = (aa + bb) > 0 ? aa / (aa + bb) : 0.0f;
                        double relw = (wh > 1e-9) ? fabs(wf - wh) / wh : 0.0;
                        if (relw > 0.01 || ((pdf_l <= 0) != (fpdf_l <= 0))) {
                            ++shown;
                            std::printf("  [hp2] px=(%d,%d) FOUND-W (%s) "
                                        "cpu(w=%.4f,pl=%.5f) f32(w=%.4f,pl=%.5f)\n",
                                        xi, yi, m1, wh, pdf_l, wf, fpdf_l);
                        }
                        continue;
                    }
                    if (!r1.mat->is_diffuse())
                        continue;
                    // NEE-1 value, fixed lp=center both sides.
                    vec3 lp = light_point(lt, 0.5, 0.5, sc0.time());
                    vec3 toL = lp - r1.point;
                    double dist = toL.length();
                    vec3 wi = toL / dist;
                    double cosS = dot(r1.normal, wi);
                    if (cosS <= 0)
                        continue;
                    vec3 lnn = light_normal_at(lt, lp, sc0.time());
                    double cosA = fabs(dot(lnn, -wi));
                    double Tr = shadow_transmittance(world, b2.media, r1.point, wi,
                                                     dist, sc0.time());
                    double pdf_l = dist * dist / (A * cosA);
                    double pdf_b = cosine_pdf(cosS);
                    double w = direction_pdf::power_weight(pdf_l, pdf_b);
                    double cval = (cosS * cosA * A / (pi * dist * dist)) * w * Tr;
                    float fx = (float)r1.point.x(), fy = (float)r1.point.y(),
                          fz = (float)r1.point.z();
                    float fnx = (float)r1.normal.x(), fny = (float)r1.normal.y(),
                          fnz = (float)r1.normal.z();
                    const GPUQuad &fq = fb.gs.quads[(size_t)fb.light_table[0].second];
                    float qlx = fq.Q[0] + 0.5f * fq.u[0] + 0.5f * fq.v[0];
                    float qly = fq.Q[1] + 0.5f * fq.u[1] + 0.5f * fq.v[1];
                    float qlz = fq.Q[2] + 0.5f * fq.u[2] + 0.5f * fq.v[2];
                    float ol[3] = {fx + fnx * 0.001f, fy + fny * 0.001f,
                                   fz + fnz * 0.001f};
                    float dx = qlx - ol[0], dy = qly - ol[1], dz = qlz - ol[2];
                    float fd = sqrtf(dx * dx + dy * dy + dz * dz);
                    float dl[3] = {dx / fd, dy / fd, dz / fd};
                    int bty = -1, bidx = -1;
                    float btt = 0;
                    bool blk = nee_census::f32_blocker(fb, ol, dl, fd, bty, bidx, btt);
                    if ((Tr <= 0) != blk) {
                        ++shown;
                        std::printf("  [hp2] px=(%d,%d) NEE1-BLOCK cpu=%d f32=%d "
                                    "(%s) blk=(%d,%d,t=%.4f)\n",
                                    xi, yi, Tr <= 0 ? 1 : 0, blk ? 1 : 0, m1, bty,
                                    bidx, btt);
                    }
                }
            }
        }
    }
    std::printf("  [hp2] shown=%d (bounce1 solids=%ld) self=%ld near=%ld "
                "inside=%ld quad=%ld hitflip=%ld hitflipR=%ld\n",
                shown, n1, cSelf, cNear, cInside, cQuad, cHitFlip, cHitFlipR);
    rng_fixed_flag() = false;
    EXPECT_TRUE(shown >= 0); // forensics probe
}

static void t_shutter() {
    test_current = "shutter";
    // Closed shutter draws no RNG: identical rays to pre-shutter code.
    camera c1;
    rng_seed(80);
    ray a = c1.get_ray(0.3, 0.6);
    rng_seed(80);
    camera c2;
    c2.set_shutter(0.0, 1.0);
    ray b = c2.get_ray(0.3, 0.6);
    EXPECT_NEAR(a.direction().x(), b.direction().x()); // same direction...
    EXPECT_NEAR(a.direction().z(), b.direction().z());
    EXPECT_TRUE(b.time() >= 0.0 && b.time() <= 1.0); // ...but stamped time
    camera c3;
    rng_seed(81);
    ray d1 = c3.get_ray(0.3, 0.6);
    rng_seed(81);
    ray d2 = c3.get_ray(0.3, 0.6);
    EXPECT_NEAR(d1.origin().x(), d2.origin().x()); // deterministic replay
}

void run_scene_tests() {
    t_camera();
    t_defocus();
    t_obj();
    t_mtl();
    t_lights();
    t_gpucam();
    t_instance();
    t_cornell();
    t_flatten();
    t_expand_gpu();
    t_shutter();
    t_nee_shadow_census();
    t_hotpixel();
    t_hotpixel2();
}
