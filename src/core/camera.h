#pragma once

#include "vec3.h"
#include "ray.h"

class camera
{
public:
    camera(vec3 lookfrom, vec3 lookat, vec3 vup, double vfov, double aspect_ratio,
           double aperture, double focus_dist)
    {
        double theta = degrees_to_radians(vfov);
        double h = std::tan(theta / 2);
        double viewport_height = 2.0 * h;
        double viewport_width = viewport_height * aspect_ratio;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
    }

    ray get_ray(double s, double t) const
    {
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

private:
    vec3 origin;
    vec3 horizontal, vertical;
    vec3 lower_left_corner;
    vec3 u, v, w;
    double lens_radius;
};