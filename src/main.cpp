#include "core/vec3.h"

#include <fstream>
#include <iostream>

int main()
{
    // vec3 v(5.0, 3.0, 2.0);
    // v + vec3(1.0, 2.0, 3.0);
    // std::cout << v.x() << ", " << v.y() << ", " << v.z() << "\n";

    vec3 v(1, 1, 1);
    v += vec3(2, 2, 2);
    std::cout << v.x() << ", " << v.y() << ", " << v.z() << "\n"; // expect 3, 3, 3

    const int width = 256;
    const int height = 256;

    std::ofstream out("output.ppm");
    out << "P3\n"
        << width << ' ' << height << "\n255\n";

    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            int r = 255 - i; // varies left to right
            int g = j;       // varies top to bottom
            int b = 128;     // constant

            out << r << ' ' << g << ' ' << b << '\n';
        }
    }

    std::cout << "Wrote output.ppm\n";
    return 0;
}