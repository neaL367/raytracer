#pragma once
// Scene builders: default NEE demo + classic Cornell box.
// main() only wires (BVH + render); all placement lives here.
// Quads double-sided, so wall winding never matters.
#include "../camera/camera.h"
#include "../core/obj_loader.h"
#include "../core/texture.h"
#include "../geometry/hittable.h"
#include "../geometry/quad.h"
#include "../geometry/sphere.h"
#include "../geometry/triangle.h"
#include "../material/material.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct scene_data {
    std::vector<std::shared_ptr<hittable>> objs;
    std::vector<std::shared_ptr<quad>> lights;
    camera cam;
};

// Axis box from 6 quads. min/max corners, one material.
inline void add_box(std::vector<std::shared_ptr<hittable>> &objs, const vec3 &lo,
                    const vec3 &hi, std::shared_ptr<material> mat) {
    double dx = hi.x() - lo.x(), dy = hi.y() - lo.y(), dz = hi.z() - lo.z();
    objs.push_back(std::make_shared<quad>(vec3(lo.x(), lo.y(), lo.z()), vec3(dx, 0, 0),
                                          vec3(0, 0, dz), mat)); // bottom
    objs.push_back(std::make_shared<quad>(vec3(lo.x(), hi.y(), lo.z()), vec3(dx, 0, 0),
                                          vec3(0, 0, dz), mat)); // top
    objs.push_back(std::make_shared<quad>(vec3(lo.x(), lo.y(), lo.z()), vec3(0, 0, dz),
                                          vec3(0, dy, 0), mat)); // x=lo
    objs.push_back(std::make_shared<quad>(vec3(hi.x(), lo.y(), lo.z()), vec3(0, 0, dz),
                                          vec3(0, dy, 0), mat)); // x=hi
    objs.push_back(std::make_shared<quad>(vec3(lo.x(), lo.y(), hi.z()), vec3(dx, 0, 0),
                                          vec3(0, dy, 0), mat)); // z=hi
    objs.push_back(std::make_shared<quad>(vec3(lo.x(), lo.y(), lo.z()), vec3(dx, 0, 0),
                                          vec3(0, dy, 0), mat)); // z=lo
}

// Moved verbatim from main (M7 scene): checker ground, cube mesh with
// sphere fallback, metal/glass spheres, ceiling light.
inline scene_data build_default(double aspect, double aperture) {
    scene_data scene;
    auto ground_mat =
        std::make_shared<lambertian>(std::make_shared<checker>(4.0, vec3(0.8, 0.8, 0.8),
                                                               vec3(0.3, 0.3, 0.3)));
    auto cube_mat =
        std::make_shared<lambertian>(std::make_shared<checker>(3.0, vec3(0.7, 0.3, 0.3),
                                                               vec3(0.9, 0.9, 0.9)));
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
        for (auto &t : mesh)
            scene.objs.push_back(t);
    }
    scene.objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, left_mat));
    scene.objs.push_back(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, right_mat));
    auto light = std::make_shared<quad>(vec3(-1, 1.9, -2), vec3(2, 0, 0),
                                        vec3(0, 0, 2), light_mat);
    scene.objs.push_back(light);
    scene.lights.push_back(light);
    scene.cam = camera(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0), 90.0, aspect,
                       aperture, 1.0);
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
    add_box(scene.objs, vec3(265, 0, 295), vec3(430, 330, 460), white); // tall
    add_box(scene.objs, vec3(130, 0, 65), vec3(295, 165, 230), white); // short

    auto light = std::make_shared<quad>(vec3(213, 554, 227), vec3(130, 0, 0),
                                        vec3(0, 0, 105), light_mat);
    scene.objs.push_back(light);
    scene.lights.push_back(light);

    vec3 from(278, 278, -800), at(278, 278, 0);
    scene.cam = camera(from, at, vec3(0, 1, 0), 40.0, aspect, aperture,
                       (from - at).length());
    return scene;
}

inline scene_data build_scene(const std::string &name, double aspect, double aperture) {
    if (name == "cornell")
        return build_cornell(aspect, aperture);
    return build_default(aspect, aperture);
}
