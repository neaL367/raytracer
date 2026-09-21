#pragma once
// Scene builders: default NEE demo + classic Cornell box.
// main() only wires (BVH + render); all placement lives here.
// Quads double-sided, so wall winding never matters.
#include "../accel/bvh.h"
#include "../accel/qbvh.h"
#include "../camera/camera.h"
#include "../core/obj_loader.h"
#include "../core/texture.h"
#include "../geometry/hittable.h"
#include "../geometry/instance.h"
#include "../geometry/light.h"
#include "../geometry/quad.h"
#include "../geometry/sphere.h"
#include "../geometry/triangle.h"
#include "../geometry/volume.h"
#include "../io/stb_loader.h"
#include "../material/material.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct scene_data {
    std::vector<std::shared_ptr<hittable>> objs;
    std::vector<light> lights; // NEE-sampled emitters (any shape)
    camera cam;
    bool env_light = false; // analytic sun+sky environment (opt-in --env)
    bool black_bg = false;  // black background for enclosed/dark scenes (book2)
};

// Axis box from 6 quads, returned as a list so callers can instance it
// (rotate/translate) or splice it flat. Corners are min/max.
inline std::shared_ptr<hittable_list> make_box_list(const vec3 &lo, const vec3 &hi,
                                                    std::shared_ptr<material> mat) {
    auto box = std::make_shared<hittable_list>();
    double dx = hi.x() - lo.x(), dy = hi.y() - lo.y(), dz = hi.z() - lo.z();
    box->add(std::make_shared<quad>(vec3(lo.x(), lo.y(), lo.z()), vec3(dx, 0, 0),
                                    vec3(0, 0, dz), mat)); // bottom
    box->add(std::make_shared<quad>(vec3(lo.x(), hi.y(), lo.z()), vec3(dx, 0, 0),
                                    vec3(0, 0, dz), mat)); // top
    box->add(std::make_shared<quad>(vec3(lo.x(), lo.y(), lo.z()), vec3(0, 0, dz),
                                    vec3(0, dy, 0), mat)); // x=lo
    box->add(std::make_shared<quad>(vec3(hi.x(), lo.y(), lo.z()), vec3(0, 0, dz),
                                    vec3(0, dy, 0), mat)); // x=hi
    box->add(std::make_shared<quad>(vec3(lo.x(), lo.y(), hi.z()), vec3(dx, 0, 0),
                                    vec3(0, dy, 0), mat)); // z=hi
    box->add(std::make_shared<quad>(vec3(lo.x(), lo.y(), lo.z()), vec3(dx, 0, 0),
                                    vec3(0, 0, dz), mat)); // z=lo
    return box;
}

// Box helper function returning hittable_list of 6 quads (Peter Shirley signature).
inline std::shared_ptr<hittable_list> box(const vec3 &lo, const vec3 &hi,
                                          std::shared_ptr<material> mat) {
    return make_box_list(lo, hi, mat);
}

// Axis box from 6 quads. min/max corners, one material.
inline void add_box(std::vector<std::shared_ptr<hittable>> &objs, const vec3 &lo,
                    const vec3 &hi, std::shared_ptr<material> mat) {
    auto box = make_box_list(lo, hi, mat);
    for (const auto &q : box->children())
        objs.push_back(q);
}

// Box by dimensions, rotated about Y then moved: the canonical Cornell pose
// (tall 15 degrees at (265,0,295), short -18 degrees at (130,0,65)).
inline std::shared_ptr<hittable> make_posed_box(const vec3 &dims, double angle_deg,
                                                const vec3 &at,
                                                std::shared_ptr<material> mat) {
    auto box = make_box_list(vec3(0, 0, 0), dims, mat);
    std::shared_ptr<hittable> posed = std::make_shared<rotate_y>(box, angle_deg);
    return std::make_shared<translate>(posed, at);
}

// Moved verbatim from main (M7 scene): checker ground, cube mesh with
// sphere fallback, metal/glass spheres, ceiling light. Shutter open =>
// left sphere drifts +0.3y (motion demo); closed => static, hashes frozen.
inline scene_data build_default(double aspect, double aperture, double sh0 = 0,
                                 double sh1 = 0, double fog_density = 0,
                                 double het_density = 0, bool marble = false,
                                 bool env = false) {
    scene_data scene;
    // Photo ground (spherical UVs); checker fallback keeps binary alive
    // when the asset is missing.
    std::shared_ptr<texture> ground_tex =
        std::make_shared<checker>(4.0, vec3(0.8, 0.8, 0.8), vec3(0.3, 0.3, 0.3));
    ppm_io::image photo;
    if (stb_loader::load_image("assets/photo_test.jpg", photo))
        // Spherical UVs over r=100: u spans 2*PI*100 world units.
        ground_tex = std::make_shared<image_texture>(photo.w, photo.h, photo.px, 628.0);
    else
        std::cerr << "assets/photo_test.jpg missing: checker ground\n";
    auto ground_mat = std::make_shared<lambertian>(ground_tex);
    // Cube takes the second photo (barycentric UVs); checker fallback.
    std::shared_ptr<texture> cube_tex =
        std::make_shared<checker>(3.0, vec3(0.7, 0.3, 0.3), vec3(0.9, 0.9, 0.9));
    ppm_io::image photo2;
    if (stb_loader::load_image("assets/photo2_test.jpg", photo2))
        // Corner UVs span one 0.7-unit cube face.
        cube_tex = std::make_shared<image_texture>(photo2.w, photo2.h, photo2.px, 0.7);
    else
        std::cerr << "assets/photo2_test.jpg missing: checker cube\n";
    auto cube_mat = std::make_shared<lambertian>(cube_tex);
    auto left_mat = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    auto right_mat = std::make_shared<dielectric>(1.5);
    auto light_mat = std::make_shared<diffuse_light>(vec3(4, 4, 4));

    scene.objs.push_back(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, ground_mat));
    std::vector<std::shared_ptr<triangle>> mesh;
    if (!obj_loader::load_obj("assets/cube.obj", mesh, cube_mat)) {
        std::cerr << "assets/cube.obj missing: falling back to sphere\n";
        scene.objs.push_back(std::make_shared<sphere>(
            vec3(0, 0, -1), 0.5, std::make_shared<lambertian>(vec3(0.7, 0.3, 0.3))));
    } else {
        for (auto &t : mesh) {
            if (sh1 > sh0) {
                // Cube drifts +0.3y with the open shutter (mesh motion demo).
                vec3 d(0, 0.3, 0);
                t->set_motion(t->vert(0) + d, t->vert(1) + d, t->vert(2) + d, sh0,
                              sh1);
            }
            scene.objs.push_back(t);
        }
    }
    if (sh1 > sh0)
        scene.objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), vec3(-1, 0.3, -1),
                                                      sh0, sh1, 0.5, left_mat));
    else
        scene.objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, left_mat));
    scene.objs.push_back(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, right_mat));
    if (marble) {
        // Opt-in marble showcase (defaults stay frozen): blue-veined sphere
        // front-left, fBm turbulence under sine banding.
        auto marble_tex = std::make_shared<noise_texture>(
            4.0, 7, 2, vec3(0.85, 0.87, 0.9), vec3(0.05, 0.15, 0.45));
        scene.objs.push_back(std::make_shared<sphere>(
            vec3(-2.2, 0.5, 0.5), 0.5, std::make_shared<lambertian>(marble_tex)));
    }
    auto light = std::make_shared<quad>(vec3(-1, 1.9, -2), vec3(2, 0, 0),
                                        vec3(0, 0, 2), light_mat);
    scene.objs.push_back(light);
    scene.lights.push_back(light);
    // Warm orb: second emitter proves multi-shape NEE (sphere sampling).
    auto orb_mat = std::make_shared<diffuse_light>(vec3(4, 2.2, 1.1));
    auto orb = std::make_shared<sphere>(vec3(2.2, 1.4, -0.6), 0.25, orb_mat);
    scene.objs.push_back(orb);
    scene.lights.push_back(orb);
    if (fog_density > 0) {
        // Smoke ball around the subject: white scatter, no NEE inside.
        auto fog_phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
        auto fog_border = std::make_shared<sphere>(vec3(0, 0, -1), 2.5, fog_phase);
        scene.objs.push_back(
            std::make_shared<constant_medium>(fog_border, fog_density, fog_phase));
    }
    if (het_density > 0) {
        // Structured smoke over the same border: sinusoidal pockets,
        // delta-tracked (unbiased). Combinable with uniform fog.
        auto het_phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
        auto het_border = std::make_shared<sphere>(vec3(0, 0, -1), 2.5, het_phase);
        scene.objs.push_back(
            std::make_shared<heterogeneous_medium>(het_border, het_density, het_phase));
    }
    scene.cam = camera(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0), 90.0, aspect,
                       aperture, 1.0);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    return scene;
}

// Classic 555 Cornell: red image-left wall, green image-right, white
// shell + 2 boxes, ceiling area light. Camera u axis points -x, so the
// x=555 wall appears image-left: red goes there for canonical look.
inline scene_data build_cornell(double aspect, double aperture, bool env = false) {
    scene_data scene;
    auto red = std::make_shared<lambertian>(vec3(0.63, 0.065, 0.05));
    auto green = std::make_shared<lambertian>(vec3(0.14, 0.45, 0.15));
    auto white = std::make_shared<lambertian>(vec3(0.725, 0.71, 0.68));
    auto light_mat = std::make_shared<diffuse_light>(vec3(7, 7, 7));

    scene.objs.push_back(std::make_shared<quad>(vec3(0, 0, 0), vec3(555, 0, 0),
                                                vec3(0, 0, 555), white)); // floor
    scene.objs.push_back(std::make_shared<quad>(vec3(0, 555, 0), vec3(555, 0, 0),
                                                vec3(0, 0, 555), white)); // ceiling
    scene.objs.push_back(std::make_shared<quad>(vec3(0, 0, 555), vec3(555, 0, 0),
                                                vec3(0, 555, 0), white)); // back
    scene.objs.push_back(std::make_shared<quad>(vec3(555, 0, 0), vec3(0, 0, 555),
                                                vec3(0, 555, 0), red)); // x=555 image-left
    scene.objs.push_back(std::make_shared<quad>(vec3(0, 0, 0), vec3(0, 0, 555),
                                                vec3(0, 555, 0), green)); // x=0 image-right
    // Posed instances: tall box 15 deg at (265,0,295), short -18 deg at
    // (130,0,65). 8 top-level objs: 5 walls + 2 instances + 1 light.
    scene.objs.push_back(
        make_posed_box(vec3(165, 330, 165), 15.0, vec3(265, 0, 295), white));
    scene.objs.push_back(
        make_posed_box(vec3(165, 165, 165), -18.0, vec3(130, 0, 65), white));

    auto light = std::make_shared<quad>(vec3(213, 554, 227), vec3(130, 0, 0),
                                        vec3(0, 0, 105), light_mat);
    scene.objs.push_back(light);
    scene.lights.push_back(light);

    vec3 from(278, 278, -800), at(278, 278, 0);
    scene.cam = camera(from, at, vec3(0, 1, 0), 40.0, aspect, aperture,
                       (from - at).length());
    scene.env_light = env;
    return scene;
}

// Classic "Ray Tracing in One Weekend" final scene: 484+ random small
// spheres + 3 hero spheres (glass, diffuse, metal) on a giant ground sphere.
inline scene_data build_weekend(double aspect, double aperture, double sh0 = 0,
                                double sh1 = 0, bool env = false) {
    scene_data scene;
    auto ground_tex =
        std::make_shared<checker>(0.32, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    auto ground_material = std::make_shared<lambertian>(ground_tex);
    scene.objs.push_back(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    scene.objs.push_back(std::make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    scene.objs.push_back(std::make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                    scene.objs.push_back(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    scene.objs.push_back(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    scene.objs.push_back(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    scene.objs.push_back(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    const double pi = 3.1415926535897932385;
    double default_aperture = 2.0 * 10.0 * std::tan((0.6 / 2.0) * pi / 180.0);
    double use_aperture = (aperture > 0.0) ? aperture : default_aperture;

    scene.cam = camera(point3(13, 2, 3), point3(0, 0, 0), vec3(0, 1, 0), 20.0, aspect,
                       use_aperture, 10.0);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    return scene;
}

// Classic "Ray Tracing: The Next Week" final scene: 400 ground boxes,
// ceiling area light, moving sphere, glass/metal spheres, subsurface and
// global fog, earth texture, procedural marble, and a rotated/translated
// cluster of 1000 spheres.
inline scene_data build_book2(double aspect, double aperture, double sh0 = 0,
                              double sh1 = 0, bool env = false) {
    scene_data scene;
    auto ground = std::make_shared<lambertian>(color(0.48, 0.83, 0.53));
    int boxes_per_side = 20;
    std::vector<std::shared_ptr<hittable>> boxes1_quads;
    boxes1_quads.reserve(boxes_per_side * boxes_per_side * 6);
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1, 101);
            auto z1 = z0 + w;

            add_box(boxes1_quads, point3(x0, y0, z0), point3(x1, y1, z1), ground);
        }
    }
    scene.objs.push_back(
        std::make_shared<qbvh_node>(boxes1_quads, 0, boxes1_quads.size(), true));

    auto light = std::make_shared<diffuse_light>(color(7, 7, 7));
    auto light_quad = std::make_shared<quad>(point3(123, 554, 147), vec3(300, 0, 0),
                                            vec3(0, 0, 265), light);
    scene.objs.push_back(light_quad);
    scene.lights.push_back(light_quad);

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    auto sphere_material = std::make_shared<lambertian>(color(0.7, 0.3, 0.1));
    scene.objs.push_back(std::make_shared<sphere>(center1, center2, 50, sphere_material));

    scene.objs.push_back(
        std::make_shared<sphere>(point3(260, 150, 45), 50, std::make_shared<dielectric>(1.5)));
    scene.objs.push_back(std::make_shared<sphere>(
        point3(0, 150, 145), 50, std::make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

    auto boundary =
        std::make_shared<sphere>(point3(360, 150, 145), 70, std::make_shared<dielectric>(1.5));
    scene.objs.push_back(boundary);
    scene.objs.push_back(std::make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));
    boundary = std::make_shared<sphere>(point3(0, 0, 0), 5000, std::make_shared<dielectric>(1.5));
    scene.objs.push_back(std::make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

    auto emat = std::make_shared<lambertian>(std::make_shared<image_texture>("earthmap.jpg"));
    scene.objs.push_back(std::make_shared<sphere>(point3(400, 200, 400), 100, emat));
    auto pertext = std::make_shared<noise_texture>(0.2);
    scene.objs.push_back(
        std::make_shared<sphere>(point3(220, 280, 300), 80, std::make_shared<lambertian>(pertext)));

    std::vector<std::shared_ptr<hittable>> boxes2_spheres;
    boxes2_spheres.reserve(1000);
    auto white = std::make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2_spheres.push_back(std::make_shared<sphere>(point3::random(0, 165), 10, white));
    }

    scene.objs.push_back(std::make_shared<translate>(
        std::make_shared<rotate_y>(
            std::make_shared<qbvh_node>(boxes2_spheres, 0, boxes2_spheres.size(), true), 15),
        vec3(-100, 270, 395)));

    point3 lookfrom(478, 278, -600);
    point3 lookat(278, 278, 0);
    vec3 vup(0, 1, 0);
    double vfov = 40.0;
    double dist_to_focus = 10.0;

    scene.cam = camera(lookfrom, lookat, vup, vfov, aspect, aperture, dist_to_focus);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    scene.black_bg = !env; // dark room unless --env explicitly requested
    return scene;
}

inline scene_data build_scene(const std::string &name, double aspect, double aperture,
                                double sh0 = 0, double sh1 = 0, double fog = 0,
                                double het = 0, bool marble = false, bool env = false) {
    if (name == "cornell")
        return build_cornell(aspect, aperture, env);
    if (name == "weekend" || name == "final" || name == "spheres" || name == "book1")
        return build_weekend(aspect, aperture, sh0, sh1, env);
    if (name == "book2" || name == "nextweek" || name == "boxes" || name == "final2")
        return build_book2(aspect, aperture, sh0, sh1, env);
    return build_default(aspect, aperture, sh0, sh1, fog, het, marble, env);
}
