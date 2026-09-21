#pragma once
// Default showcase scene: photo ground + photo cube + brushed metal +
// glass + quad light + warm orb, optional marble/fog/het smoke.
// main() only wires (BVH + render); all placement lives here.
#include "common.h"

// Moved verbatim from main (M7 scene): checker ground, cube mesh with
// sphere fallback, metal/glass spheres, ceiling light. Shutter open =>
// left sphere drifts +0.3y (motion demo); closed => static, hashes frozen.
inline scene_data build_default(double aspect, double aperture, double sh0 = 0,
                                double sh1 = 0, double fog_density = 0,
                                double het_density = 0, bool marble = false,
                                bool env = false) {
    scene_data scene;
    // ---- Materials: photo ground (spherical UVs) + photo cube ----
    // Checker fallbacks keep the binary alive when assets are missing.
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
    auto left_mat = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.15, 0.5);
    auto right_mat = std::make_shared<dielectric>(1.5);
    auto light_mat = std::make_shared<diffuse_light>(vec3(4, 4, 4));

    // ---- Objects: ground + mesh + hero spheres ----
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
    // ---- Lights: ceiling quad + warm orb (multi-shape NEE demo) ----
    auto light = std::make_shared<quad>(vec3(-1, 1.9, -2), vec3(2, 0, 0),
                                        vec3(0, 0, 2), light_mat);
    scene.objs.push_back(light);
    scene.lights.push_back(light);
    // Warm orb: second emitter proves multi-shape NEE (sphere sampling).
    auto orb_mat = std::make_shared<diffuse_light>(vec3(4, 2.2, 1.1));
    auto orb = std::make_shared<sphere>(vec3(2.2, 1.4, -0.6), 0.25, orb_mat);
    scene.objs.push_back(orb);
    scene.lights.push_back(orb);
    // ---- Volumes: uniform fog + structured hetero smoke (opt-in flags) ----
    if (fog_density > 0) {
        // Smoke ball around the subject: white scatter, no NEE inside.
        auto fog_phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
        auto fog_border = std::make_shared<sphere>(vec3(0, 0, -1), 2.5, fog_phase);
        auto fog = std::make_shared<constant_medium>(fog_border, fog_density, fog_phase);
        scene.objs.push_back(fog);
        scene.media.push_back(fog);
    }
    if (het_density > 0) {
        // Structured smoke over the same border: sinusoidal pockets,
        // delta-tracked (unbiased). Combinable with uniform fog.
        auto het_phase = std::make_shared<isotropic>(vec3(0.9, 0.9, 0.9));
        auto het_border = std::make_shared<sphere>(vec3(0, 0, -1), 2.5, het_phase);
        auto het = std::make_shared<heterogeneous_medium>(het_border, het_density, het_phase);
        scene.objs.push_back(het);
        scene.media.push_back(het);
    }
    // ---- Camera + environment ----
    scene.cam = camera(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0), 90.0, aspect,
                       aperture, 1.0);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    return scene;
}
