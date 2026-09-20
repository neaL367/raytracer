#pragma once
// OBJ: v positions, vn normals, f faces (v, v/vt, v//vn, v/vt/vn).
// Fan-triangulates polygons, handles relative (-) indices. vt parsed
// and ignored (no image-UV mapping on meshes yet: flagged). Winding
// irrelevant: triangles are double-sided via set_face_normal.
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

// Corner reference: position + optional normal indices (-1 = absent).
struct corner {
    int v = -1;
    int n = -1;
};

inline bool parse_corner(const std::string &tok, size_t nv, size_t nn, corner &c) {
    // Forms: v | v/vt | v//vn | v/vt/vn. vt dropped deliberately.
    size_t s1 = tok.find('/');
    std::string vs = (s1 == std::string::npos) ? tok : tok.substr(0, s1);
    if (vs.empty())
        return false;
    c.v = fix_index(std::stoi(vs), nv);
    if (s1 != std::string::npos) {
        size_t s2 = tok.find('/', s1 + 1);
        std::string ns = (s2 == std::string::npos) ? "" : tok.substr(s2 + 1);
        if (!ns.empty())
            c.n = fix_index(std::stoi(ns), nn);
    }
    return c.v >= 0;
}

inline bool load_obj(const std::string &path, std::vector<std::shared_ptr<triangle>> &tris,
                     std::shared_ptr<material> mat) {
    std::ifstream f(path);
    if (!f)
        return false;
    std::vector<vec3> verts, normals;
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
        } else if (tag == "vn") {
            double x, y, z;
            if (ls >> x >> y >> z)
                normals.emplace_back(x, y, z);
        } else if (tag == "f") {
            std::vector<corner> idx;
            std::string tok;
            while (ls >> tok) {
                corner c;
                if (parse_corner(tok, verts.size(), normals.size(), c))
                    idx.push_back(c);
            }
            if (idx.size() < 3)
                continue;
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                const corner &a = idx[0], &b = idx[i], &cc = idx[i + 1];
                if ((size_t)a.v >= verts.size() || (size_t)b.v >= verts.size() ||
                    (size_t)cc.v >= verts.size())
                    continue;
                bool smooth = a.n >= 0 && b.n >= 0 && cc.n >= 0 &&
                              (size_t)a.n < normals.size() && (size_t)b.n < normals.size() &&
                              (size_t)cc.n < normals.size();
                if (smooth)
                    tris.push_back(std::make_shared<triangle>(
                        verts[(size_t)a.v], verts[(size_t)b.v], verts[(size_t)cc.v],
                        unit_vector(normals[(size_t)a.n]), unit_vector(normals[(size_t)b.n]),
                        unit_vector(normals[(size_t)cc.n]), mat));
                else
                    tris.push_back(std::make_shared<triangle>(
                        verts[(size_t)a.v], verts[(size_t)b.v], verts[(size_t)cc.v], mat));
            }
        }
    }
    return !tris.empty();
}

} // namespace obj_loader
