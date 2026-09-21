#pragma once
// Scene builders: default NEE demo + classic Cornell box.
// main() only wires (BVH + render); all placement lives here.
// Quads double-sided, so wall winding never matters.
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
                                    vec3(0, dy, 0), mat)); // z=lo
    return box;
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
                                double het_density = 0) {
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
    return scene;
}

// Classic 555 Cornell: red image-left wall, green image-right, white
// shell + 2 boxes, ceiling area light. Camera u axis points -x, so the
// x=555 wall appears image-left: red goes there for canonical look.
inline scene_data build_cornell(double aspect, double aperture) {
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
    return scene;
}

inline scene_data build_scene(const std::string &name, double aspect, double aperture,
                               double sh0 = 0, double sh1 = 0, double fog = 0,
                               double het = 0) {
    if (name == "cornell")
        return build_cornell(aspect, aperture);
    return build_default(aspect, aperture, sh0, sh1, fog, het);
}
