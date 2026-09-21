#pragma once
// Classic 555 Cornell: red image-left wall, green image-right, white
// shell + 2 posed boxes, ceiling area light. Camera u axis points -x, so the
// x=555 wall appears image-left: red goes there for canonical look.
#include "common.h"

inline scene_data build_cornell(double aspect, double aperture, bool env = false) {
    scene_data scene;
    // ---- Materials: book-canonical albedos ----
    auto red = std::make_shared<lambertian>(vec3(0.63, 0.065, 0.05));
    auto green = std::make_shared<lambertian>(vec3(0.14, 0.45, 0.15));
    auto white = std::make_shared<lambertian>(vec3(0.725, 0.71, 0.68));
    auto light_mat = std::make_shared<diffuse_light>(vec3(7, 7, 7));

    // ---- Objects: shell walls + posed boxes ----
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

    // ---- Lights: ceiling area quad ----
    auto light = std::make_shared<quad>(vec3(213, 554, 227), vec3(130, 0, 0),
                                        vec3(0, 0, 105), light_mat);
    scene.objs.push_back(light);
    scene.lights.push_back(light);

    // ---- Camera + environment ----
    vec3 from(278, 278, -800), at(278, 278, 0);
    scene.cam = camera(from, at, vec3(0, 1, 0), 40.0, aspect, aperture,
                       (from - at).length());
    scene.env_light = env;
    return scene;
}
