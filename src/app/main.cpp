#include "core/vec3.h"
#include "core/ray.h"
#include "camera/camera.h"
#include "geometry/sphere.h"
#include "output/ppm.h"

#include <filesystem>
#include <iostream>
#include <vector>

// Normal shade for M1: maps surface orientation to RGB directly.
// Reason: hit vs miss instantly visible, no material system yet.
vec3 ray_color(const ray &r, const sphere &s) {
    hit_record rec;
    if (s.hit(r, 0.001, 1e30, rec))
        return 0.5 * (rec.normal + vec3(1, 1, 1));
    vec3 unit = unit_vector(r.direction());
    double t = 0.5 * (unit.y() + 1.0);
    return (1.0 - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0);
}

int main() {
    const int W = 400;
    const int H = static_cast<int>(W / (16.0 / 9.0));

    camera cam;
    sphere s(vec3(0, 0, -1), 0.5);

    std::vector<vec3> fb(W * H);
    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            double u = double(i) / (W - 1);
            double v = double(j) / (H - 1);
            fb[j * W + i] = ray_color(cam.get_ray(u, v), s);
        }
    }

    std::filesystem::create_directories("out");
    if (!write_ppm("out/image.ppm", fb, W, H)) {
        std::cerr << "write failed\n";
        return 1;
    }
    std::cout << "wrote out/image.ppm " << W << "x" << H << "\n";
    return 0;
}
