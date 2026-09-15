#include "core/vec3.h"
#include "core/ray.h"
#include "core/sphere.h"
#include "core/hittable_list.h"
#include "core/camera.h"
#include "core/random.h"
#include "core/triangle.h"
#include "core/obj_loader.h"
#include "core/bvh.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>

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
    auto start = std::chrono::high_resolution_clock::now();

    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / (16.0 / 9.0));
    const int samples_per_pixel = 200;
    const int max_depth = 50;

    // --- build a FLAT list of every individual primitive, no nested hittable_lists ---
    hittable_list flat_objects;

    auto material_ground = std::make_shared<lambertian>(vec3(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(vec3(1.0, 0.0, 0.0));
    auto material_right = std::make_shared<metal>(vec3(0.8, 0.6, 0.2));
    auto material_triangle = std::make_shared<lambertian>(vec3(0.2, 0.8, 0.2));
    auto material_mesh = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));

    auto mesh = load_obj("assets/model.obj", material_mesh);
    std::cout << "Loaded " << mesh->size() << " triangles\n";

    // flatten the mesh's triangles directly into flat_objects, instead of nesting the list
    for (const auto &tri : mesh->objects_ref())
        flat_objects.add(tri);

    flat_objects.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, material_ground));
    flat_objects.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, material_center));
    flat_objects.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, material_right));
    flat_objects.add(std::make_shared<triangle>(
        vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2),
        material_triangle));

    // --- scale the scene up to a size where a BVH actually pays off ---
    for (int i = 0; i < 300; ++i)
    {
        vec3 center(random_double(-10, 10), random_double(-10, 10), random_double(-15, -5));
        flat_objects.add(std::make_shared<sphere>(center, 0.2, material_mesh));
    }

    std::cout << "Total flat primitives: " << flat_objects.size() << "\n";

    // --- build the BVH once, from the fully flattened list ---
    hittable_list bvh_world;
    bvh_world.add(std::make_shared<bvh_node>(flat_objects));

    vec3 lookfrom(4, 3, 5);
    vec3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    double dist_to_focus = (lookfrom - lookat).length();
    double aperture = 0.05;

    camera cam(lookfrom, lookat, vup, 25, 16.0 / 9.0, aperture, dist_to_focus);

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
                pixel_color += ray_color(r, bvh_world, max_depth); // through the BVH now
                // pixel_color += ray_color(r, flat_objects, max_depth); // temporarily bypass the BVH
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
    }

    std::cout << "Wrote output.ppm\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Render time: " << elapsed.count() << " seconds\n";

    return 0;
}