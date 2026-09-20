#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/sphere.h"
#include "geometry/triangle.h"
#include "material/material.h"
#include "output/ppm.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Recursive path stub M2: attenuation chain to depth limit, sky on miss.
// Depth 10 stops glass ping-pong. tMin kills acne. NEE/RR wait till M4.
vec3 ray_color(const ray &r, const hittable &world, int depth) {
    if (depth <= 0)
        return vec3(0, 0, 0);
    hit_record rec;
    if (world.hit(r, 0.001, 1e30, rec)) {
        ray scattered;
        vec3 attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return vec3(0, 0, 0);
    }
    vec3 unit = unit_vector(r.direction());
    double t = 0.5 * (unit.y() + 1.0);
    return (1.0 - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0);
}

int main(int argc, char **argv) {
    int spp = 16; // 4x4 strata default; perfect squares stratify
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--samples")
            spp = std::max(1, std::atoi(argv[i + 1]));

    const int W = 400;
    const int H = static_cast<int>(W / (16.0 / 9.0));
    const int max_depth = 10;
    const unsigned base_seed = 42;

    hittable_list world;
    auto ground_mat = std::make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
    auto center_mat = std::make_shared<lambertian>(vec3(0.7, 0.3, 0.3));
    auto left_mat = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    auto right_mat = std::make_shared<dielectric>(1.5);

    world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, ground_mat));
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, center_mat));
    world.add(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, left_mat));
    world.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, right_mat));
    world.add(std::make_shared<triangle>(vec3(-0.3, -0.35, -0.6), vec3(0.3, -0.35, -0.6),
                                         vec3(0, 0.1, -0.6), center_mat));

    camera cam;
    std::vector<vec3> fb(W * H);
    if (spp == 1)
        rng_seed(base_seed); // legacy M2 stream: bit-identical to M2
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            // Reseed per pixel: same seed -> same image regardless of
            // later thread scheduling (threads land in M5).
            if (spp != 1)
                rng_seed(base_seed + (unsigned)(j * W + i));
            vec3 acc(0, 0, 0);
            if (spp == 1) {
                // M2 path verbatim: edge-inclusive mapping, no jitter.
                double u = double(i) / (W - 1);
                double v = double(j) / (H - 1);
                acc = ray_color(cam.get_ray(u, v), world, max_depth);
            } else {
                auto offs = pixel_samples(spp);
                for (auto [ox, oy] : offs) {
                    double u = (i + ox) / W;
                    double v = (j + oy) / H;
                    acc += ray_color(cam.get_ray(u, v), world, max_depth);
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
              << " depth=" << max_depth << "\n";
    return 0;
}
