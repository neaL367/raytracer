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
}
