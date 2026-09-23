#pragma once
// Milk-glass SSS demo (M57): nested dielectric shells + interior scattering
// medium, assembled from existing parts. Outer glass (pri 1) contains an
// inner milk boundary (pri 2) holding a constant medium: rays refract
// air->glass->milk, random-walk, and exit, giving SSS glow with zero new
// transport. Dark room (black_bg) + ceiling quad light, book2-style.
#include "common.h"

inline scene_data build_sss(double aspect, double aperture) {
    scene_data scene;
    // ---- Ground: big lambertian slab top ----
    auto ground = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    scene.objs.push_back(std::make_shared<quad>(point3(-400, 0, -400), vec3(800, 0, 0),
                                                vec3(0, 0, 800), ground));
    // ---- Light: ceiling area quad ----
    auto light = std::make_shared<diffuse_light>(color(7, 7, 7));
    auto light_quad = std::make_shared<quad>(point3(-60, 300, -40), vec3(120, 0, 0),
                                             vec3(0, 0, 120), light);
    scene.objs.push_back(light_quad);
    scene.lights.push_back(light_quad);

    // ---- Milk-glass ball: shell pri 1, milk boundary pri 2, medium ----
    point3 center(0, 70, 0);
    auto shell = std::make_shared<sphere>(center, 70, std::make_shared<dielectric>(1.5, 0.0, 1));
    auto milk_skin =
        std::make_shared<sphere>(center, 50, std::make_shared<dielectric>(1.33, 0.0, 2));
    auto milk =
        std::make_shared<constant_medium>(milk_skin, 0.06, color(0.97, 0.97, 0.94));
    scene.objs.push_back(shell);
    scene.objs.push_back(milk_skin);
    scene.objs.push_back(milk);
    scene.media.push_back(milk);

    // ---- Camera + environment ----
    point3 lookfrom(0, 160, -330);
    point3 lookat(0, 90, 0);
    vec3 vup(0, 1, 0);
    double vfov = 40.0;
    scene.cam = camera(lookfrom, lookat, vup, vfov, aspect, aperture, 10.0);
    scene.black_bg = true; // dark room: SSS glow reads on black
    return scene;
}
