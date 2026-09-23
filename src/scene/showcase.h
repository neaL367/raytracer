#pragma once
// Masterpiece showcase scene demonstrating the full power of the raytracer:
// 1. Anisotropic GGX conductor (brushed gold with UV tangents)
// 2. Optical dielectric with concentric hollow bubble (TIR, caustic focus)
// 3. Tinted dielectric with Beer's law volumetric absorption (amber glass)
// 4. Procedural 3D marble (multi-octave Perlin fBm with sine vein banding)
// 5. Subsurface scattering (nested translucent medium inside clear skin)
// 6. Heterogeneous turbulent volume (3D sinusoidal Pocket smoke)
// 7. OBJ triangle mesh with texture mapping (cube.obj + photo texture)
// 8. Kinetic motion blur (linear shutter motion between sh0 and sh1)
// 9. Posed architectural pedestals (rotate_y + translate instances)
// 10. Multi-emitter NEE (studio quad key + warm orb + neon cyan rim)
// 11. Environment sun+sky support (opt-in --env)
// Default resolution: 1200x675 landscape (adaptive framing for portraits as well).
#include "common.h"

inline scene_data build_showcase(double aspect, double aperture, double sh0 = 0,
                                 double sh1 = 0, double fog_density = 0,
                                 double het_density = 0, bool marble_opt = false,
                                 bool env = false) {
    scene_data scene;

    // =========================================================================
    // 1. Materials & Textures
    // =========================================================================

    // Stage floor: Rich dark/light checker tile
    auto floor_tex = std::make_shared<checker>(
        2.0, vec3(0.07, 0.08, 0.09), vec3(0.82, 0.84, 0.86));
    auto floor_mat = std::make_shared<lambertian>(floor_tex);

    // Architectural pedestals: Smooth architectural slate
    auto pedestal_mat = std::make_shared<lambertian>(vec3(0.18, 0.20, 0.22));
    auto rear_stage_mat = std::make_shared<lambertian>(vec3(0.12, 0.13, 0.15));

    // Hero 1: Anisotropic brushed gold/brass conductor
    // Tight along u (0.05), streaked along v (0.38)
    auto aniso_gold = std::make_shared<metal>(vec3(0.95, 0.82, 0.46), 0.05, 0.38);

    // Hero 2: Optical glass (IOR = 1.5)
    auto glass_mat = std::make_shared<dielectric>(1.5);

    // Hero 3: Absorbing tinted amber glass (Beer-Lambert law along exit chords)
    auto amber_glass = std::make_shared<dielectric>(1.52, 0.0, 1, vec3(0.18, 0.72, 1.85));

    // Hero 4: Procedural 3D marble (Perlin fBm turbulence + sine vein banding)
    auto marble_tex = std::make_shared<noise_texture>(
        4.0, 7, 2, vec3(0.88, 0.90, 0.94), vec3(0.06, 0.16, 0.42));
    auto marble_mat = std::make_shared<lambertian>(marble_tex);

    // Hero 5: Subsurface scattering phase function
    auto sss_phase = std::make_shared<isotropic>(vec3(0.92, 0.95, 0.90));

    // Hero 6: Heterogeneous smoke phase function
    auto het_phase = std::make_shared<isotropic>(vec3(0.95, 0.95, 0.95));

    // Hero 7: Textured OBJ cube mesh
    std::shared_ptr<texture> mesh_tex =
        std::make_shared<checker>(3.0, vec3(0.8, 0.25, 0.25), vec3(0.95, 0.95, 0.95));
    ppm_io::image photo;
    if (stb_loader::load_image("assets/photo2_test.jpg", photo))
        mesh_tex = std::make_shared<image_texture>(photo.w, photo.h, photo.px, 0.7);
    else if (stb_loader::load_image("assets/photo_test.jpg", photo))
        mesh_tex = std::make_shared<image_texture>(photo.w, photo.h, photo.px, 0.7);
    auto mesh_mat = std::make_shared<lambertian>(mesh_tex);

    // Hero 8: Kinetic polished chrome metal
    auto chrome_mat = std::make_shared<metal>(vec3(0.95, 0.97, 1.0), 0.04);

    // Lighting rig materials
    auto key_light_mat = std::make_shared<diffuse_light>(vec3(7.5, 7.2, 6.8));
    auto orb_light_mat = std::make_shared<diffuse_light>(vec3(6.5, 3.2, 1.2));
    auto rim_light_mat = std::make_shared<diffuse_light>(vec3(1.5, 4.5, 9.5));

    // =========================================================================
    // 2. Stage & Architectural Pedestals
    // =========================================================================

    // Large stage floor extending into the studio space at y = -0.5
    scene.objs.push_back(std::make_shared<quad>(
        vec3(-15, -0.5, -15), vec3(30, 0, 0), vec3(0, 0, 30), floor_mat));

    // Left pedestal: Posed box rotated 15 degrees, base at y = -0.5, height 0.65
    scene.objs.push_back(make_posed_box(vec3(0.9, 0.65, 0.9), 15.0,
                                        vec3(-2.05, -0.5, -1.05), pedestal_mat));

    // Right pedestal: Posed box rotated -15 degrees, base at y = -0.5, height 0.65
    scene.objs.push_back(make_posed_box(vec3(0.9, 0.65, 0.9), -15.0,
                                        vec3(1.15, -0.5, -1.05), pedestal_mat));

    // Rear gallery plinth: Wide platform, base at y = -0.5, height 0.95
    scene.objs.push_back(make_posed_box(vec3(3.6, 0.95, 1.1), 0.0,
                                        vec3(-1.8, -0.5, -3.15), rear_stage_mat));

    // =========================================================================
    // 3. Hero Objects on Display
    // =========================================================================

    // --- LEFT PEDESTAL: Brushed Anisotropic Gold Conductor ---
    scene.objs.push_back(std::make_shared<sphere>(
        vec3(-1.6, 0.15 + 0.45, -0.6), 0.45, aniso_gold));

    // --- RIGHT PEDESTAL: Procedural 3D Marble Sphere ---
    scene.objs.push_back(std::make_shared<sphere>(
        vec3(1.6, 0.15 + 0.45, -0.6), 0.45, marble_mat));

    // --- CENTER: Textured OBJ Cube Mesh ---
    std::vector<std::shared_ptr<triangle>> mesh;
    if (obj_loader::load_obj("assets/cube.obj", mesh, mesh_mat)) {
        for (auto &t : mesh) {
            if (sh1 > sh0) {
                vec3 md(0, 0.2, 0);
                t->set_motion(t->vert(0) + md, t->vert(1) + md, t->vert(2) + md, sh0, sh1);
            }
            scene.objs.push_back(t);
        }
    } else {
        scene.objs.push_back(std::make_shared<sphere>(
            vec3(0.0, -0.15, -1.0), 0.35, mesh_mat));
    }

    // --- CENTER FOREGROUND: Hollow Optical Glass Bubble ---
    // Outer glass bubble (r = 0.40, IOR = 1.5)
    vec3 bubble_center(0.0, -0.1, 0.1);
    scene.objs.push_back(std::make_shared<sphere>(bubble_center, 0.40, glass_mat));
    // Inner negative radius sphere creates concentric air cavity for TIR
    scene.objs.push_back(std::make_shared<sphere>(bubble_center, -0.32, glass_mat));

    // --- FOREGROUND LEFT: Absorbing Amber Glass (Beer's Law) ---
    scene.objs.push_back(std::make_shared<sphere>(
        vec3(-0.95, -0.15, 0.5), 0.35, amber_glass));

    // --- REAR GALLERY LEFT: Subsurface Scattering (SSS) Jade/Milk Sphere ---
    vec3 sss_center(-1.1, 0.45 + 0.45, -2.4);
    double sss_dens = (fog_density > 0) ? fog_density : 0.65;
    auto sss_border = std::make_shared<sphere>(sss_center, 0.45, sss_phase);
    auto sss_medium = std::make_shared<constant_medium>(sss_border, sss_dens, sss_phase);
    scene.objs.push_back(sss_medium);
    scene.media.push_back(sss_medium);

    // --- REAR GALLERY RIGHT: Heterogeneous Turbulent Smoke Pocket ---
    vec3 het_center(1.1, 0.45 + 0.45, -2.4);
    double het_dens = (het_density > 0) ? het_density : 0.75;
    auto het_border = std::make_shared<sphere>(het_center, 0.45, het_phase);
    auto het_medium = std::make_shared<heterogeneous_medium>(het_border, het_dens, het_phase);
    scene.objs.push_back(het_medium);
    scene.media.push_back(het_medium);

    // --- KINETIC SATELLITE: Motion-Blurred Floating Orb ---
    vec3 m_start(0.0, 1.45, -1.1);
    vec3 m_end = (sh1 > sh0) ? vec3(0.2, 1.65, -0.9) : m_start;
    if (sh1 > sh0) {
        scene.objs.push_back(std::make_shared<sphere>(
            m_start, m_end, sh0, sh1, 0.25, chrome_mat));
    } else {
        scene.objs.push_back(std::make_shared<sphere>(m_start, 0.25, chrome_mat));
    }

    // Opt-in extra marble sphere for test suite compatibility
    if (marble_opt) {
        scene.objs.push_back(std::make_shared<sphere>(
            vec3(-2.6, -0.1, 0.2), 0.4, marble_mat));
    }

    // =========================================================================
    // 4. Multi-Emitter Lighting Rig (NEE with MIS)
    // =========================================================================

    // 1. Soft Overhead Studio Key Light (Quad emitter)
    auto key_light = std::make_shared<quad>(
        vec3(-2.4, 3.8, -3.2), vec3(4.8, 0, 0), vec3(0, 0, 3.8), key_light_mat);
    scene.objs.push_back(key_light);
    scene.lights.push_back(key_light);

    // 2. Warm Glowing Orb (Spherical emitter)
    auto warm_orb = std::make_shared<sphere>(
        vec3(2.2, 1.35, -0.6), 0.28, orb_light_mat);
    scene.objs.push_back(warm_orb);
    scene.lights.push_back(warm_orb);

    // 3. Neon Cyan Rim Light (Slender vertical quad emitter)
    auto rim_light = std::make_shared<quad>(
        vec3(-2.7, 0.1, -2.6), vec3(0, 2.6, 0), vec3(0.2, 0, 0.4), rim_light_mat);
    scene.objs.push_back(rim_light);
    scene.lights.push_back(rim_light);

    // =========================================================================
    // 5. Camera & Environment
    // =========================================================================

    // Adaptive vertical FOV:
    // Aspect < 1.0 (e.g. 600x1200 portrait): vfov widens to 52 deg to frame vertically.
    // Aspect >= 1.0 (e.g. 1200x600 landscape): vfov 38 deg to frame horizontally.
    double vfov = (aspect < 1.0) ? 52.0 : 38.0;

    vec3 lookfrom(0.0, 1.6, 3.8);
    vec3 lookat(0.0, 0.35, -0.8);
    vec3 vup(0.0, 1.0, 0.0);
    double focus_dist = (lookfrom - lookat).length();

    scene.cam = camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    scene.black_bg = !env; // Sleek dark studio void backdrop

    return scene;
}
