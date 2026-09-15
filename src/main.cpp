#include "core/vec3.h"
#include "core/ray.h"
#include "core/sphere.h"
#include "core/hittable_list.h"
#include "core/camera.h"
#include "core/random.h"
#include "core/triangle.h"
#include "core/obj_loader.h"

#include <fstream>
#include <iostream>
#include <memory>

vec3 ray_color(const ray &r, const hittable &world, int depth)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    hit_record rec;
    if (world.hit(r, 0.001, 1000.0, rec))
    {
        ray scattered;
        vec3 attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered))
        {
            return attenuation * ray_color(scattered, world, depth - 1);
        }
        return vec3(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}

int main()
{
    // image
    const int image_width = 720;
    const int image_height = static_cast<int>(image_width / (16.0 / 9.0));
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    // world
    hittable_list world;

    // auto material_behind_glass = std::make_shared<lambertian>(vec3(1.0, 1.0, 0.0));
    auto material_ground = std::make_shared<lambertian>(vec3(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(vec3(1.0, 0.0, 0.0));
    // auto material_left = std::make_shared<dielectric>(1.5);
    auto material_right = std::make_shared<metal>(vec3(0.8, 0.6, 0.2));
    auto material_triangle = std::make_shared<lambertian>(vec3(0.2, 0.8, 0.2));

    auto material_mesh = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));
    auto mesh = load_obj("assets/model.obj", material_mesh);
    std::cout << "Loaded " << mesh->size() << " triangles\n";
    world.add(mesh);

    // world.add(std::make_shared<sphere>(vec3(-2.5, 0, -2.5), 0.6, material_behind_glass));
    world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, material_ground));
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, material_center));
    // world.add(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, material_left));
    world.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, material_right));
    world.add(std::make_shared<triangle>(
        vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2),
        material_triangle));

    // camera
    vec3 lookfrom(4, 3, 5);
    vec3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    double dist_to_focus = (lookfrom - lookat).length();
    double aperture = 0.05; // subtle — we want to see the cube's edges sharply, not blurred

    camera cam(lookfrom, lookat, vup, 25, 16.0 / 9.0, aperture, dist_to_focus);

    // render
    std::ofstream out("output.ppm");
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            vec3 pixel_color(0, 0, 0);

            for (int sample = 0; sample < samples_per_pixel; ++sample)
            {
                double s = (i + random_double()) / (image_width - 1);
                double t = (j + random_double()) / (image_height - 1);

                ray r = cam.get_ray(s, t);
                pixel_color += ray_color(r, world, max_depth);
            }

            double scale = 1.0 / samples_per_pixel;
            double r_out = std::sqrt(pixel_color.x() * scale);
            double g_out = std::sqrt(pixel_color.y() * scale);
            double b_out = std::sqrt(pixel_color.z() * scale);

            int ir = static_cast<int>(255.999 * r_out);
            int ig = static_cast<int>(255.999 * g_out);
            int ib = static_cast<int>(255.999 * b_out);

            out << ir << ' ' << ig << ' ' << ib << '\n';
        }

        // vec3 v(5.0, 3.0, 2.0);
        // v + vec3(1.0, 2.0, 3.0);
        // std::cout << v.x() << ", " << v.y() << ", " << v.z() << "\n";

        // vec3 v(1, 1, 1);
        // v += vec3(2, 2, 2);
        // std::cout << v.x() << ", " << v.y() << ", " << v.z() << "\n";

        // vec3 v(6, 5, 5);
        // std::cout << "v = " << v << "\n";

        // vec3 u(1,2,1);
        // vec3 v(2,1,2);

        // dot(u, v);
        // cross(u, v);

        // std::cout << "Dot product of u and v: " << dot(u, v) << "\n";
        // std::cout << "Cross product of u and v: (" << cross(u, v).x() << ", " << cross(u, v).y() << ", " << cross(u, v).z() << ")\n";

        // vec3(3,4,0).length();
        // unit_vector(vec3(3,4,0)).length();

        // std::cout << "Length of (3,4,0): " << vec3(3,4,0).length() << "\n";
        // std::cout << "Length of unit vector of (3,4,0): " << unit_vector(vec3(3,4,0)).length() << "\n";

        // ray r(vec3(0, 0, 0), vec3(1, 0, 0));
        // std::cout << "at t=0: " << r.at(0) << "\n"; // expect 0 0 0
        // std::cout << "at t=2: " << r.at(2) << "\n"; // expect 2 0 0

        // sphere s(vec3(0, 0, -1), 0.5);
        // ray r(vec3(0, 0, 0), vec3(0, 0, -1));
        // hit_record rec;
        // if (s.hit(r, 0.001, 1000.0, rec))
        // {
        //     std::cout << "Hit at t=" << rec.t << ", point=" << rec.point << ", normal=" << rec.normal << "\n";
        // }
        // else
        // {
        //     std::cout << "No hit\n";
        // }

        // hittable_list world;
        // world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5));
        // world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100));

        // ray r(vec3(0, 0, 0), vec3(0, 0, -1));
        // hit_record rec;
        // if (world.hit(r, 0.001, 1000.0, rec))
        // {
        //     std::cout << "Hit at t=" << rec.t << "\n";
        // }
    }

    std::cout << "Wrote output.ppm\n";
    return 0;
}