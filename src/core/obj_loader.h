#pragma once
// Minimal OBJ: v positions + f indices only. Fan-triangulates polygons,
// handles relative (-) indices. No normals/uvs/groups yet (flagged).
// Winding irrelevant: triangles are double-sided via set_face_normal.
#include "geometry/triangle.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace obj_loader {

inline int fix_index(int idx, size_t n) {
    if (idx > 0)
        return idx - 1;
    return (int)n + idx; // negative = relative to end
}

inline bool load_obj(const std::string &path, std::vector<std::shared_ptr<triangle>> &tris,
                     std::shared_ptr<material> mat) {
    std::ifstream f(path);
    if (!f)
        return false;
    std::vector<vec3> verts;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag) || tag[0] == '#')
            continue;
        if (tag == "v") {
            double x, y, z;
            if (ls >> x >> y >> z)
                verts.emplace_back(x, y, z);
        } else if (tag == "f") {
            std::vector<int> idx;
            std::string tok;
            while (ls >> tok) {
                // Take leading int before any '/'.
                size_t slash = tok.find('/');
                std::string num = (slash == std::string::npos) ? tok : tok.substr(0, slash);
                if (num.empty() || num == "-0")
                    continue;
                int v = std::stoi(num);
                idx.push_back(fix_index(v, verts.size()));
            }
            if (idx.size() < 3)
                continue;
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                int a = idx[0], b = idx[i], c = idx[i + 1];
                if (a < 0 || b < 0 || c < 0 || (size_t)a >= verts.size() ||
                    (size_t)b >= verts.size() || (size_t)c >= verts.size())
                    continue;
                tris.push_back(std::make_shared<triangle>(verts[(size_t)a], verts[(size_t)b],
                                                          verts[(size_t)c], mat));
            }
        }
    }
    return !tris.empty();
}

} // namespace obj_loader
