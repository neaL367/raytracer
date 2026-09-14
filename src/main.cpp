#include "core/vec3.h"

#include <fstream>
#include <iostream>

int main()
{
    vec3 v(5.0, 3.0, 2.0);

    vec3 operator_plus = v + vec3(1.0, 2.0, 3.0);
    std::cout << operator_plus.x() << ", " << operator_plus.y() << ", " << operator_plus.z() << "\n";
    
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