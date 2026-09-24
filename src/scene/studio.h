#pragma once
// Studio Showcase Scene: architectural cylinder columns, pedestal disks,
// overhead disk area lighting, and metallic/glass display spheres.
#include "common.h"
#include "../geometry/disk.h"
#include "../geometry/cylinder.h"

inline scene_data build_studio(double aspect, double aperture,
                               double sh0 = 0, double sh1 = 0,
                               double fog = 0, double het = 0,
                               bool marble = false, bool env = false) {
    (void)sh0; (void)sh1; (void)fog; (void)het; (void)marble; (void)env;
    scene_data s;

    // Materials
    auto mat_floor = std::make_shared<lambertian>(vec3(0.2, 0.22, 0.25));
    auto mat_back = std::make_shared<lambertian>(vec3(0.08, 0.09, 0.11));
    auto mat_gold = std::make_shared<metal>(vec3(0.95, 0.78, 0.35), 0.08);
    auto mat_chrome = std::make_shared<metal>(vec3(0.9, 0.9, 0.92), 0.02);
    auto mat_glass = std::make_shared<dielectric>(1.52, 0.025); // Cauchy glass
    auto mat_bubble = std::make_shared<dielectric>(1.05, 0.0, 0, vec3(0, 0, 0), true);
    auto mat_ruby = std::make_shared<dielectric>(1.77, 0.01, 0, vec3(0.1, 0.8, 0.7));
    auto mat_pedestal = std::make_shared<lambertian>(vec3(0.12, 0.12, 0.14));
    auto mat_column = std::make_shared<lambertian>(vec3(0.75, 0.75, 0.78));

    // Studio Area Lights
    auto mat_overhead_light = std::make_shared<diffuse_light>(vec3(12.0, 11.5, 10.0));
    auto mat_rim_light = std::make_shared<diffuse_light>(vec3(4.0, 6.0, 10.0));

    // 1. Studio Floor & Curved Cyclorama Backdrop
    s.objs.push_back(std::make_shared<quad>(vec3(-10, 0, -10), vec3(20, 0, 0), vec3(0, 0, 20), mat_floor));
    s.objs.push_back(std::make_shared<quad>(vec3(-10, 0, -6), vec3(20, 0, 0), vec3(0, 10, 0), mat_back));

    // 2. Overhead Circular Disk Area Light (soft circular highlights)
    auto top_disk_light = std::make_shared<disk>(vec3(0, 4.5, 0), vec3(0, -1, 0), 1.2, mat_overhead_light);
    s.objs.push_back(top_disk_light);
    s.lights.push_back(light(top_disk_light));

    // Rim Disk Light (cool blue fill from side)
    auto rim_disk_light = std::make_shared<disk>(vec3(3.5, 2.5, -2.0), vec3(-1, -0.3, 0.5), 0.8, mat_rim_light);
    s.objs.push_back(rim_disk_light);
    s.lights.push_back(light(rim_disk_light));

    // 3. Central Tiered Pedestal (Disks + Cylinder)
    s.objs.push_back(std::make_shared<disk>(vec3(0, 0.005, 0), vec3(0, 1, 0), 2.2, mat_pedestal)); // Base platform
    s.objs.push_back(std::make_shared<cylinder>(vec3(0, 0, 0), vec3(0, 0.45, 0), 1.5, mat_pedestal));
    s.objs.push_back(std::make_shared<disk>(vec3(0, 0.455, 0), vec3(0, 1, 0), 1.6, mat_gold, 1.4)); // Gold outer ring

    // 4. Architectural Fluted Columns (Cylinders) in background
    double col_rad = 0.28;
    s.objs.push_back(std::make_shared<cylinder>(vec3(-2.8, 0, -3.5), vec3(-2.8, 4.0, -3.5), col_rad, mat_column));
    s.objs.push_back(std::make_shared<cylinder>(vec3(-1.4, 0, -4.5), vec3(-1.4, 4.0, -4.5), col_rad, mat_column));
    s.objs.push_back(std::make_shared<cylinder>(vec3( 1.4, 0, -4.5), vec3( 1.4, 4.0, -4.5), col_rad, mat_column));
    s.objs.push_back(std::make_shared<cylinder>(vec3( 2.8, 0, -3.5), vec3( 2.8, 4.0, -3.5), col_rad, mat_column));

    // 5. Display Objects on Pedestal
    // Centerpiece Glass Sphere with Cauchy Dispersion
    s.objs.push_back(std::make_shared<sphere>(vec3(0.0, 1.05, 0.0), 0.60, mat_glass));
    // Ruby Gemstone Sphere on Left
    s.objs.push_back(std::make_shared<sphere>(vec3(-0.95, 0.78, 0.4), 0.33, mat_ruby));
    // Polished Mirror Chrome Sphere on Right
    s.objs.push_back(std::make_shared<sphere>(vec3(0.95, 0.80, 0.35), 0.35, mat_chrome));
    // Thin-film Soap Bubble Floating above
    s.objs.push_back(std::make_shared<sphere>(vec3(-0.35, 1.95, -0.2), 0.28, mat_bubble));

    // 6. Camera Setup
    vec3 lookfrom(0.0, 1.4, 3.2);
    vec3 lookat(0.0, 0.95, 0.0);
    double vfov = 36.0;
    double dist_to_focus = (lookfrom - lookat).length();
    s.cam = camera(lookfrom, lookat, vec3(0, 1, 0), vfov, aspect, aperture, dist_to_focus);
    s.black_bg = true;
    s.env_light = false;

    return s;
}
