#pragma once
// Shared scene plumbing: data layout, box helpers, posed-box helper.
// Per-scene builders live in showcase.h / default.h;
// scene.h re-exports them behind the stable build_scene() dispatcher.
// Quads double-sided, so wall winding never matters.
#include "../accel/bvh.h"
#include "../accel/qbvh.h"
#include "../camera/camera.h"
#include "../core/obj_loader.h"
#include "../core/random.h"
#include "../core/texture.h"
#include "../core/vec3.h"
#include "../geometry/hittable.h"
#include "../geometry/instance.h"
#include "../geometry/light.h"
#include "../geometry/quad.h"
#include "../geometry/sphere.h"
#include "../geometry/triangle.h"
#include "../geometry/volume.h"
#include "../io/stb_loader.h"
#include "../material/material.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../core/hdri.h"

struct scene_data {
    std::vector<std::shared_ptr<hittable>> objs;
    std::vector<light> lights; // NEE-sampled emitters (any shape)
    std::vector<std::shared_ptr<hittable>> media; // volumes for NEE transmittance
    camera cam;
    bool env_light = false; // analytic sun+sky environment (opt-in --env)
    bool black_bg = false; // black background for enclosed/dark scenes (book2)
    // M63: HDRI env map (nenv_mode: 0=off, 1=analytic, 2=HDRI).
    std::shared_ptr<hdri_env> hdri; // null when not loaded
    int nenv_mode() const {
        if (hdri && !hdri->empty()) return 2;
        return env_light ? 1 : 0;
    }
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

// Showcase catalog for --list-scenes (name, aliases, default output).
inline void print_scenes() {
    std::cout << "scenes (name [aliases] default WxH@spp):\n"
              << "  showcase [show demo] 1200x675@512 all engine powers (aniso, glass, SSS, smoke, marble, mesh, multi-light)\n"
              << "  default           400x225@16  photo ground, cube, metal, glass, quad, orb\n";
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
