#pragma once
// Classic "Ray Tracing in One Weekend" final scene: 484+ random small
// spheres + 3 hero spheres (glass, diffuse, metal) on a giant ground sphere.
// NOTE: small-sphere placement is RNG-driven, so CPU and GPU builds (and
// repeat runs) place them differently by construction; parity here is
// statistical (means), never bit-exact.
#include "common.h"

inline scene_data build_weekend(double aspect, double aperture, double sh0 = 0,
                                double sh1 = 0, bool env = false) {
    scene_data scene;
    // ---- Materials: checkerboard ground ----
    auto ground_tex =
        std::make_shared<checker>(0.32, color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    auto ground_material = std::make_shared<lambertian>(ground_tex);
    scene.objs.push_back(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    // ---- Objects: random small spheres (book recipe) ----
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

    // ---- Objects: 3 hero spheres ----
    auto material1 = std::make_shared<dielectric>(1.5);
    scene.objs.push_back(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    scene.objs.push_back(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    scene.objs.push_back(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    // ---- Camera + environment: vfov-20 telephoto, defocus lens ----
    const double pi = 3.1415926535897932385;
    double default_aperture = 2.0 * 10.0 * std::tan((0.6 / 2.0) * pi / 180.0);
    double use_aperture = (aperture > 0.0) ? aperture : default_aperture;

    scene.cam = camera(point3(13, 2, 3), point3(0, 0, 0), vec3(0, 1, 0), 20.0, aspect,
                       use_aperture, 10.0);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    return scene;
}
