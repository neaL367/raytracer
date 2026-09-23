#pragma once
// Masterpiece showcase scene demonstrating the full power of the raytracer:
// 1. Benchmark 3D Mesh: Stanford Bunny with smooth vertex normals and anisotropic gold
// 2. Thin-Film Interference (M68): Physical Airy soap bubble with Newton rings
// 3. Cauchy Spectral Dispersion (M67): Prismatic crystal sphere with rainbow caustics
// 4. Physical Conductor (M67): Measured n/k copper conductor with thin-film sheen
// 5. Procedural 3D Marble: Multi-octave Perlin fBm with sine vein banding
// 6. Rough Dielectric (M43): Frosted emerald glass with GGX BTDF transmission
// 7. Beer's Law Absorbing Glass (M58): Deep amber glass along exit chords
// 8. Subsurface Scattering (SSS): Translucent jade medium inside clear boundary
// 9. Heterogeneous Turbulent Volume: 3D sinusoidal Pocket smoke
// 10. Kinetic Motion Blur: Shutter motion satellite orb
// 11. Studio Cyclorama: Curved seamless architectural studio stage
// 12. Multi-Emitter NEE: Studio key softbox + warm orb + rim light
#include "common.h"
#include "../core/spectrum.h"

inline scene_data build_showcase(double aspect, double aperture, double sh0 = 0,
                                 double sh1 = 0, double fog_density = 0,
                                 double het_density = 0, bool marble_opt = false,
                                 bool env = false) {
    scene_data scene;

    // =========================================================================
    // 1. Materials & Textures
    // =========================================================================

    // Studio Cyclorama Floor: Architectural slate tile
    auto floor_tex = std::make_shared<checker>(
        2.5, vec3(0.09, 0.10, 0.12), vec3(0.16, 0.17, 0.20));
    auto floor_mat = std::make_shared<lambertian>(floor_tex);

    // Studio Cyclorama Wall: Deep neutral studio backdrop
    auto wall_mat = std::make_shared<lambertian>(vec3(0.12, 0.13, 0.15));

    // Architectural Pedestals: Dark slate and obsidian plinths
    auto pedestal_mat = std::make_shared<lambertian>(vec3(0.15, 0.16, 0.18));
    auto rear_stage_mat = std::make_shared<lambertian>(vec3(0.10, 0.11, 0.13));

    // Centerpiece: Brushed Anisotropic Gold Conductor
    // Tight along u (0.04), streaked along v (0.32)
    auto bunny_mat = std::make_shared<metal>(vec3(0.95, 0.82, 0.46), 0.04, 0.32);

    // Hero 1: Iridescent Soap Bubble (M68 Airy Thin-Film Interference)
    // Physical soap film (d = 420nm, n = 1.33) yielding vibrant Newton rings
    auto soap_mat = std::make_shared<dielectric>(1.33);
    soap_mat->set_film(420.0, 1.33);

    // Hero 2: Optical Crystal with Cauchy Dispersion (M67 Spectral Hero)
    // Cauchy B = 0.022 um^2 produces vivid chromatic rainbow dispersion
    auto crystal_mat = std::make_shared<dielectric>(1.54, 0.0, 1, vec3(0, 0, 0), 0.022);

    // Hero 3: Physical Copper Conductor (M67 Conductor n/k preset 3)
    // Plus subtle oxide thin-film sheen (d = 160nm, n = 2.1)
    auto copper_mat = std::make_shared<metal>(3, 0.05, 0.05);
    copper_mat->set_film(160.0, 2.1);

    // Hero 4: Procedural 3D Veined Marble (Perlin fBm turbulence)
    auto marble_tex = std::make_shared<noise_texture>(
        4.0, 7, 2, vec3(0.92, 0.94, 0.96), vec3(0.08, 0.18, 0.46));
    auto marble_mat = std::make_shared<lambertian>(marble_tex);

    // Hero 5: Absorbing Amber Glass (M58 Beer-Lambert law)
    auto amber_glass = std::make_shared<dielectric>(1.52, 0.0, 1, vec3(0.20, 0.80, 2.20));

    // Hero 6: Rough Frosted Emerald Glass (M43 GGX BTDF transmission roughness)
    auto rough_glass = std::make_shared<dielectric>(1.52, 0.12, 1, vec3(0.10, 0.45, 0.25));

    // Hero 7: Subsurface Scattering (SSS) Jade Medium
    auto sss_phase = std::make_shared<isotropic>(vec3(0.48, 0.92, 0.65));

    // Hero 8: Heterogeneous Turbulent Smoke Pocket
    auto het_phase = std::make_shared<isotropic>(vec3(0.92, 0.92, 0.95));

    // Hero 9: Kinetic polished chrome metal
    auto chrome_mat = std::make_shared<metal>(vec3(0.96, 0.98, 1.0), 0.02);

    // Studio Lighting Rig: high-CRI softboxes
    auto key_light_mat = std::make_shared<diffuse_light>(vec3(7.6, 7.3, 6.8));
    auto orb_light_mat = std::make_shared<diffuse_light>(vec3(6.5, 3.8, 1.6));
    auto rim_light_mat = std::make_shared<diffuse_light>(vec3(2.5, 4.2, 7.5));

    // =========================================================================
    // 2. Studio Cyclorama & Architectural Pedestals
    // =========================================================================

    // Stage floor at y = -0.5
    scene.objs.push_back(std::make_shared<quad>(
        vec3(-15, -0.5, -5.0), vec3(30, 0, 0), vec3(0, 0, 22.0), floor_mat));

    // Seamless studio cyclorama back wall
    scene.objs.push_back(std::make_shared<quad>(
        vec3(-15, -0.5, -4.2), vec3(30, 0, 0), vec3(0, 9.0, 0), wall_mat));

    // Center pedestal: Base at y = -0.5, height 0.65, top at y = 0.15
    scene.objs.push_back(make_posed_box(vec3(1.15, 0.65, 1.15), 0.0,
                                        vec3(-0.575, -0.5, -1.225), pedestal_mat));

    // Left pedestal: Posed box rotated 15 degrees, base at y = -0.5, height 0.65
    scene.objs.push_back(make_posed_box(vec3(0.9, 0.65, 0.9), 15.0,
                                        vec3(-2.15, -0.5, -1.15), pedestal_mat));

    // Right pedestal: Posed box rotated -15 degrees, base at y = -0.5, height 0.65
    scene.objs.push_back(make_posed_box(vec3(0.9, 0.65, 0.9), -15.0,
                                        vec3(1.25, -0.5, -1.15), pedestal_mat));

    // Rear gallery plinth: Wide platform, base at y = -0.5, height 0.95
    scene.objs.push_back(make_posed_box(vec3(4.2, 0.95, 1.1), 0.0,
                                        vec3(-2.1, -0.5, -3.05), rear_stage_mat));

    // =========================================================================
    // 3. Hero Objects on Display
    // =========================================================================

    // --- CENTERPIECE: Stanford Bunny with Smooth Normals & Anisotropic Gold ---
    std::vector<std::shared_ptr<triangle>> mesh;
    if (obj_loader::load_obj("assets/bunny.obj", mesh, bunny_mat)) {
        for (auto &t : mesh) {
            if (sh1 > sh0) {
                vec3 md(0, 0.05, 0);
                t->set_motion(t->vert(0) + md, t->vert(1) + md, t->vert(2) + md, sh0, sh1);
            }
            scene.objs.push_back(t);
        }
    } else {
        scene.objs.push_back(std::make_shared<sphere>(
            vec3(0.0, 0.50, -0.65), 0.40, bunny_mat));
    }

    // --- FRONT LEFT HERO: Iridescent Thin-Film Soap Bubble ---
    vec3 bubble_center(-1.10, -0.05, 0.35);
    scene.objs.push_back(std::make_shared<sphere>(bubble_center, 0.36, soap_mat));
    scene.objs.push_back(std::make_shared<sphere>(bubble_center, -0.355, soap_mat));

    // --- FRONT RIGHT HERO: Cauchy Dispersive Crystal Sphere ---
    vec3 crystal_center(1.10, -0.05, 0.35);
    scene.objs.push_back(std::make_shared<sphere>(crystal_center, 0.36, crystal_mat));
    scene.objs.push_back(std::make_shared<sphere>(crystal_center, -0.25, crystal_mat));

    // --- FOREGROUND CENTER: Absorbing Amber Glass (Beer's Law) ---
    scene.objs.push_back(std::make_shared<sphere>(
        vec3(0.0, -0.16, 0.85), 0.30, amber_glass));

    // --- LEFT PEDESTAL: Physical Copper Conductor (M67 n/k) ---
    scene.objs.push_back(std::make_shared<sphere>(
        vec3(-1.75, 0.65, -0.7), 0.45, copper_mat));

    // --- RIGHT PEDESTAL: Procedural 3D Veined Marble Sphere ---
    scene.objs.push_back(std::make_shared<sphere>(
        vec3(1.75, 0.65, -0.7), 0.45, marble_mat));

    // --- REAR GALLERY LEFT: Subsurface Scattering (SSS) Jade Sphere ---
    vec3 sss_center(-1.25, 0.90, -2.4);
    double sss_dens = (fog_density > 0) ? fog_density : 0.70;
    auto sss_border = std::make_shared<sphere>(sss_center, 0.45, sss_phase);
    auto sss_medium = std::make_shared<constant_medium>(sss_border, sss_dens, sss_phase);
    scene.objs.push_back(sss_medium);
    scene.media.push_back(sss_medium);

    // --- REAR GALLERY RIGHT: Rough Frosted Emerald Glass (GGX BTDF) ---
    vec3 rough_center(1.25, 0.90, -2.4);
    scene.objs.push_back(std::make_shared<sphere>(rough_center, 0.45, rough_glass));

    // --- REAR GALLERY CENTER: Heterogeneous Turbulent Smoke Pocket ---
    vec3 het_center(0.0, 0.90, -2.6);
    double het_dens = (het_density > 0) ? het_density : 0.80;
    auto het_border = std::make_shared<sphere>(het_center, 0.42, het_phase);
    auto het_medium = std::make_shared<heterogeneous_medium>(het_border, het_dens, het_phase);
    scene.objs.push_back(het_medium);
    scene.media.push_back(het_medium);

    // --- KINETIC SATELLITE: Motion-Blurred Floating Orb ---
    vec3 m_start(0.15, 1.55, -0.8);
    vec3 m_end = (sh1 > sh0) ? vec3(0.30, 1.65, -0.7) : m_start;
    if (sh1 > sh0) {
        scene.objs.push_back(std::make_shared<sphere>(
            m_start, m_end, sh0, sh1, 0.20, chrome_mat));
    } else {
        scene.objs.push_back(std::make_shared<sphere>(m_start, 0.20, chrome_mat));
    }

    // Opt-in extra marble sphere for test suite compatibility
    if (marble_opt) {
        scene.objs.push_back(std::make_shared<sphere>(
            vec3(-2.6, -0.1, 0.2), 0.4, marble_mat));
    }

    // =========================================================================
    // 4. Multi-Emitter Studio Lighting Rig (NEE with MIS)
    // =========================================================================

    // 1. Soft Overhead Studio Key Softbox (Quad emitter, angled 45 deg down)
    auto key_light = std::make_shared<quad>(
        vec3(-3.0, 4.8, -2.0), vec3(6.0, 0, 0), vec3(0, -1.5, 3.2), key_light_mat);
    scene.objs.push_back(key_light);
    scene.lights.push_back(key_light);

    // 2. Warm Glowing Orb (Spherical fill emitter, off-stage right)
    auto warm_orb = std::make_shared<sphere>(
        vec3(3.6, 2.2, -0.5), 0.28, orb_light_mat);
    scene.objs.push_back(warm_orb);
    scene.lights.push_back(warm_orb);

    // 3. Neon Cyan Rim Light (Recessed accent quad behind the gallery)
    auto rim_light = std::make_shared<quad>(
        vec3(-4.8, 0.4, -3.2), vec3(0, 3.0, 0), vec3(0.4, 0, 0.8), rim_light_mat);
    scene.objs.push_back(rim_light);
    scene.lights.push_back(rim_light);

    // =========================================================================
    // 5. Camera & Studio Environment
    // =========================================================================

    double vfov = (aspect < 1.0) ? 52.0 : 37.0;

    vec3 lookfrom(0.0, 1.55, 3.8);
    vec3 lookat(0.0, 0.42, -0.65);
    vec3 vup(0.0, 1.0, 0.0);
    double focus_dist = (lookfrom - lookat).length();

    scene.cam = camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    scene.cam.set_shutter(sh0, sh1);
    scene.env_light = env;
    scene.black_bg = !env; // Studio cyclorama captures light; void is enclosed

    return scene;
}
