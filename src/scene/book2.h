#pragma once
// Classic "Ray Tracing: The Next Week" final scene: 400 ground boxes,
// ceiling area light, moving sphere, glass/metal spheres, subsurface and
// global fog, earth texture, procedural marble, and a rotated/translated
// cluster of 1000 spheres.
#include "common.h"

inline scene_data build_book2(double aspect, double aperture, double sh0 = 0,
                              double sh1 = 0, bool env = false) {
    scene_data scene;
    // ---- Materials ----
    auto ground = std::make_shared<lambertian>(color(0.48, 0.83, 0.53));

    // ---- Objects: 400 ground boxes under one BVH ----
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

    // ---- Lights: ceiling area quad ----
    auto light = std::make_shared<diffuse_light>(color(7, 7, 7));
    auto light_quad = std::make_shared<quad>(point3(123, 554, 147), vec3(300, 0, 0),
                                            vec3(0, 0, 265), light);
    scene.objs.push_back(light_quad);
    scene.lights.push_back(light_quad);

    // ---- Objects: hero spheres (moving + glass + metal) ----
    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    auto sphere_material = std::make_shared<lambertian>(color(0.7, 0.3, 0.1));
    scene.objs.push_back(std::make_shared<sphere>(center1, center2, 50, sphere_material));

    scene.objs.push_back(
        std::make_shared<sphere>(point3(260, 150, 45), 50, std::make_shared<dielectric>(1.5)));
    scene.objs.push_back(std::make_shared<sphere>(
        point3(0, 150, 145), 50, std::make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

    // ---- Volumes: blue subsurface ball + global white haze ----
    // NOTE: registered in scene.media so NEE transmittance accounts them.
    auto boundary =
        std::make_shared<sphere>(point3(360, 150, 145), 70, std::make_shared<dielectric>(1.5));
    auto blue_smoke =
        std::make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9));
    scene.objs.push_back(boundary);
    scene.objs.push_back(blue_smoke);
    scene.media.push_back(blue_smoke);
    boundary = std::make_shared<sphere>(point3(0, 0, 0), 5000, std::make_shared<dielectric>(1.5));
    auto haze = std::make_shared<constant_medium>(boundary, .0001, color(1, 1, 1));
    scene.objs.push_back(haze);
    scene.media.push_back(haze);

    // ---- Objects: textured + procedural spheres ----
    auto emat = std::make_shared<lambertian>(std::make_shared<image_texture>("earthmap.jpg"));
    scene.objs.push_back(std::make_shared<sphere>(point3(400, 200, 400), 100, emat));
    auto pertext = std::make_shared<noise_texture>(0.2);
    scene.objs.push_back(
        std::make_shared<sphere>(point3(220, 280, 300), 80, std::make_shared<lambertian>(pertext)));

    // ---- Objects: rotated/translated white-sphere cluster ----
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

    // ---- Camera + environment: dark room unless --env ----
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
