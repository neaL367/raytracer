#include <fstream>
#include <iostream>

int main() {
    const int width = 256;
    const int height = 256;

    std::ofstream out("output.ppm");
    out << "P3\n" << width << ' ' << height << "\n255\n";

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int r = 255 - i;      // varies left to right
            int g = j;      // varies top to bottom
            int b = 128;    // constant

            out << r << ' ' << g << ' ' << b << '\n';
        }
    }

    std::cout << "Wrote output.ppm\n";
    return 0;
}