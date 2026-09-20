#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/sphere.h"
#include "geometry/triangle.h"
#include "geometry/quad.h"
#include "material/material.h"
#include "integrator/integrator.h"
#include "output/ppm.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    int spp = 16; // 4x4 strata default; perfect squares stratify
    double aperture = 0.0; // 0 = pinhole (M3 path); >0 thin-lens blur
    for (int i = 1; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if (a == "--samples")
            spp = std::max(1, std::atoi(argv[i + 1]));
        else if (a == "--aperture")
            aperture = std::max(0.0, std::atof(argv[i + 1]));
    }

    const int W = 400;
    const int H = static_cast<int>(W / (16.0 / 9.0));
    const int max_depth = 50; // RR handles termination; depth is backstop
    const unsigned base_seed = 42;

    hittable_list world;
    std::vector<std::shared_ptr<quad>> lights;
    auto ground_mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    auto center_mat = std::make_shared<lambertian>(vec3(0.7, 0.3, 0.3));
    auto left_mat = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    auto right_mat = std::make_shared<dielectric>(1.5);
    auto light_mat = std::make_shared<diffuse_light>(vec3(4, 4, 4));

    world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, ground_mat));
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, center_mat));
    world.add(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, left_mat));
    world.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, right_mat));
    world.add(std::make_shared<triangle>(vec3(-0.3, -0.35, -0.6), vec3(0.3, -0.35, -0.6),
                                         vec3(0, 0.1, -0.6), center_mat));
    // Ceiling area light, faces down into scene.
    auto light = std::make_shared<quad>(vec3(-1, 1.9, -2), vec3(2, 0, 0),
                                        vec3(0, 0, 2), light_mat);
    world.add(light);
    lights.push_back(light);

    // Focus at sphere plane (dist 1); aperture 0 reproduces pinhole.
    camera cam(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0),
               90.0, double(W) / double(H), aperture, 1.0);
    integrator tracer;
    std::vector<vec3> fb(W * H);
    if (spp == 1)
        rng_seed(base_seed); // legacy stream path
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            if (spp != 1)
                rng_seed(base_seed + (unsigned)(j * W + i));
            vec3 acc(0, 0, 0);
            if (spp == 1) {
                double u = double(i) / (W - 1);
                double v = double(j) / (H - 1);
                acc = tracer.Li(cam.get_ray(u, v), world, lights, max_depth);
            } else {
                auto offs = pixel_samples(spp);
                for (auto [ox, oy] : offs) {
                    double u = (i + ox) / W;
                    double v = (j + oy) / H;
                    acc += tracer.Li(cam.get_ray(u, v), world, lights, max_depth);
                }
                acc /= (double)offs.size();
            }
            fb[j * W + i] = acc;
        }
    }

    std::filesystem::create_directories("out");
    if (!write_ppm("out/image.ppm", fb, W, H)) {
        std::cerr << "write failed\n";
        return 1;
    }
    std::cout << "wrote out/image.ppm " << W << "x" << H << " spp=" << spp
              << " depth=" << max_depth << " aperture=" << aperture << "\n";
    return 0;
}
