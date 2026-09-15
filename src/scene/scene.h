#pragma once
// Scene assembly shared by the CPU renderer and the GPU uploader: one
// construction order, one RNG stream, zero drift between backends. Returns
// the flat primitive list (BVH input / SoA upload source) plus the light
// list for direct-light estimation.
#include "../app/config.h"
#include "../core/camera.h"
#include "../core/hittable_list.h"
#include "../core/material.h"
#include "../core/obj_loader.h"
#include "../core/quad.h"
#include "../core/random.h"
#include "../core/sphere.h"
#include "../core/texture.h"
#include "../core/triangle.h"
#include "../core/vec3.h"

#include <iostream>
#include <memory>
#include <vector>

struct scene_data
{
    hittable_list objects; // flat: every primitive individually, no nesting
    std::vector<std::shared_ptr<quad>> lights;
    // Non-null only with --ground outside: tested beside the tree, not in it.
    std::shared_ptr<hittable> ground_outside;
};

// The one camera both backends use. Aperture comes from config so parity
// runs can stop the lens down to a pinhole (identical primary rays).
inline camera default_camera(double aperture)
{
    vec3 lookfrom(4, 3, 5);
    vec3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    double dist_to_focus = (lookfrom - lookat).length();
    return camera(lookfrom, lookat, vup, 25, 16.0 / 9.0, aperture, dist_to_focus);
}

inline scene_data build_scene(const render_config &cfg)
{
    scene_data scene;

    auto material_ground = std::make_shared<lambertian>(
        std::make_shared<checker_texture>(0.32, vec3(0.8, 0.8, 0.8), vec3(0.2, 0.2, 0.2)));
    auto material_center = std::make_shared<lambertian>(
        std::make_shared<image_texture>("assets/uv_check.ppm", cfg.do_bilinear));
    auto material_right = std::make_shared<metal>(vec3(0.8, 0.6, 0.2));
    auto material_triangle = std::make_shared<lambertian>(vec3(0.2, 0.8, 0.2));
    auto material_mesh = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));

    auto mesh = load_obj("assets/model.obj", material_mesh);
    std::cout << "Loaded " << mesh->size() << " triangles\n";

    // flatten the mesh's triangles directly, instead of nesting the list
    for (const auto &tri : mesh->objects_ref())
        scene.objects.add(tri);

    scene.objects.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, material_center));
    scene.objects.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, material_right));
    if (cfg.use_glass)
    {
        // Deliberately clear of the center sphere (1.33 apart, radii sum 1.0)
        // so the parity scene has no interpenetration to argue about.
        scene.objects.add(std::make_shared<sphere>(vec3(-1.2, 0.3, -0.5), 0.5,
                                                   std::make_shared<dielectric>(1.5)));
    }
    scene.objects.add(std::make_shared<triangle>(
        vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2),
        material_triangle));

    // Overhead area light. Lives outside the object list unless NEE is on, so
    // the default path renders the exact historical scene (anchor hash holds).
    if (cfg.do_nee)
    {
        auto light_mat = std::make_shared<diffuse_light>(vec3(3, 3, 3));
        auto area_light = std::make_shared<quad>(
            vec3(-2, 5, -2), vec3(4, 0, 0), vec3(0, 0, 4), light_mat);
        scene.lights.push_back(area_light);
        scene.objects.add(area_light);
        // Cool fill against the warm key: exercises multi-light mixture
        // (uniform choice, 1/n weights, averaged pdfs) on every backend.
        auto fill_mat = std::make_shared<diffuse_light>(vec3(1.5, 1.8, 2.5));
        auto fill_light = std::make_shared<quad>(
            vec3(3, 3.5, 1), vec3(2, 0, 0), vec3(0, 0, 2), fill_mat);
        scene.lights.push_back(fill_light);
        scene.objects.add(fill_light);
    }

    // --- scale the scene up to a size where a BVH actually pays off ---
    for (int i = 0; i < cfg.extra_spheres; ++i)
    {
        vec3 center(random_double(-10, 10), random_double(-10, 10), random_double(-15, -5));
        scene.objects.add(std::make_shared<sphere>(center, 0.2, material_mesh));
    }

    // The ground is a radius-100 sphere whose box overlaps nearly everything.
    // Inside the tree it poisons every ancestor box it touches; tested
    // separately the tree only holds finite objects. Closest-hit logic makes
    // both placements render identically — this flag measures the cost gap.
    auto ground = std::make_shared<sphere>(vec3(0, -100.5, -1), 100, material_ground);
    if (cfg.ground_in_bvh)
        scene.objects.add(ground);
    else
        scene.ground_outside = ground;

    std::cout << "Total flat primitives: " << scene.objects.size() << "\n";
    return scene;
}
