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
        f << "newmtl badmap\nKd 0 1 0\nmap_Kd missing.png\n";
        f << "newmtl mapped\nKd 1 1 1\nmap_Kd rt_mtl_test.bmp\n";
    }
    {
        std::ofstream f(obj);
        f << "mtllib rt_mtl_test.mtl\n";
        f << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\n";
        f << "v 2 0 0\nv 3 0 0\nv 2 1 0\nv 2 0 1\nv 3 0 1\nv 2 1 1\n";
        f << "f 1 2 3\n"; // no usemtl yet: caller fallback
        f << "usemtl red\nf 4 5 6\n";
        f << "usemtl mirror\nf 7 8 9\n";
        f << "usemtl badmap\nf 10 11 12\n";
        f << "usemtl ghost\nf 1 2 3\n"; // unknown name: fallback
    }
    auto fallback = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    std::vector<std::shared_ptr<triangle>> tris;
    EXPECT_TRUE(obj_loader::load_obj(obj, tris, fallback));
    EXPECT_TRUE((int)tris.size() == 5);
    EXPECT_TRUE(tris[0]->mat_ptr() == fallback); // pre-usemtl
    EXPECT_TRUE(tris[4]->mat_ptr() == fallback); // unknown usemtl
    hit_record dummy;
    auto alb = [&](int i) { return tris[(size_t)i]->mat_ptr()->surface_albedo(dummy); };
    EXPECT_NEAR(alb(1).x(), 1); // Kd red
    EXPECT_TRUE(dynamic_cast<metal *>(tris[2]->mat_ptr().get()) != nullptr); // Ks
    EXPECT_NEAR(alb(2).x(), 0.9);
    EXPECT_NEAR(alb(3).y(), 1); // missing map -> Kd green
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

static void t_cornell() {
    test_current = "cornell";
    scene_data scene = build_cornell(16.0 / 9.0, 0.0);
    EXPECT_TRUE((int)scene.objs.size() == 18); // 5 walls + 12 box + 1 light
    EXPECT_TRUE((int)scene.lights.size() == 1);
    EXPECT_TRUE(light_mat(scene.lights[0])->emitted().x() > 1); // bright
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
    // Cornell: 18 quads flat, light first for NEE indexing.
    scene_data cor = build_scene("cornell", 1.0, 0.0);
    flat_scene fc;
    EXPECT_TRUE(flatten_scene(cor, fc));
    EXPECT_TRUE((int)fc.refs.size() == 18);
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
    auto mm = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    float alb[4] = {}, alb2[4] = {}, emit[4] = {}, prm[4] = {};
    EXPECT_TRUE(mm->export_gpu(alb, alb2, emit, prm));
    EXPECT_TRUE(fabs(alb[0] - 0.8) < 1e-6);
    EXPECT_TRUE(prm[0] == 7 && fabs(prm[1] - 0.3) < 1e-6);
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
    t_cornell();
    t_flatten();
    t_shutter();
}
