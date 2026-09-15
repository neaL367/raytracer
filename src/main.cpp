#include "core/vec3.h"
#include "core/ray.h"
#include "core/sphere.h"
#include "core/hittable.h"
#include "core/hittable_list.h"

#include <fstream>
#include <iostream>
#include <memory>

vec3 ray_color(const ray &r, const hittable &world)
{
    hit_record rec;
    if (world.hit(r, 0.0, 1000.0, rec))
    {
        return 0.5 * (rec.normal + vec3(1, 1, 1));
    }

    // background: simple vertical gradient, blue to white
    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}

int main()
{
    // image
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // world
    hittable_list world;
    world.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100));

    // camera
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);
    double focal_length = 1.0;

    vec3 origin(0, 0, 0);
    vec3 horizontal(viewport_width, 0, 0);
    vec3 vertical(0, viewport_height, 0);
    vec3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    // render
    std::ofstream out("output.ppm");
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            double u = static_cast<double>(i) / (image_width - 1);
            double v = static_cast<double>(j) / (image_height - 1);

            ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
            vec3 pixel_color = ray_color(r, world);

            int ir = static_cast<int>(255.999 * pixel_color.x());
            int ig = static_cast<int>(255.999 * pixel_color.y());
            int ib = static_cast<int>(255.999 * pixel_color.z());

            out << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::cout << "Wrote output.ppm\n";
    return 0;

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