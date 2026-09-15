#pragma once
#include "vec3.h"
#include "triangle.h"
#include "hittable_list.h"
#include "material.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <iostream>

inline std::shared_ptr<hittable_list> load_obj(const std::string &filepath, std::shared_ptr<material> mat)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open OBJ file: " << filepath << "\n";
        return std::make_shared<hittable_list>();
    }

    std::vector<vec3> vertices;
    auto mesh = std::make_shared<hittable_list>();

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            double x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(vec3(x, y, z));
        }
        else if (prefix == "f")
        {
            std::vector<int> indices;
            std::string token;

            while (iss >> token)
            {
                int idx = std::stoi(token); // handles "5" or "5/2/1" by stopping at the first '/'
                indices.push_back(idx - 1); // OBJ indices are 1-based; convert to 0-based
            }

            for (size_t i = 1; i + 1 < indices.size(); ++i)
            {
                mesh->add(std::make_shared<triangle>(
                    vertices[indices[0]],
                    vertices[indices[i]],
                    vertices[indices[i + 1]],
                    mat));
            }
        }
    }

    return mesh;
}