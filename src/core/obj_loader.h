#pragma once
// OBJ: v positions, vn normals, vt uvs, f faces (v, v/vt, v//vn,
// v/vt/vn). Fan-triangulates polygons, handles relative (-) indices.
// Winding irrelevant: triangles are double-sided via set_face_normal.
// MTL: mtllib + usemtl wires per-face materials (Kd / Ks / Ns / map_Kd).
// Faces without usemtl keep the caller fallback. Unknown statements
// (illum, d, Ke, ...) warn once per file, never silently.
// MTL/map paths resolve relative to their own file's directory.
#include "geometry/triangle.h"
#include "material/material.h"
#include "io/stb_loader.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace obj_loader {

inline std::string file_dir(const std::string &path) {
    size_t p = path.find_last_of("/\\");
    return (p == std::string::npos) ? "" : path.substr(0, p + 1);
}

inline vec3 clamp01(const vec3 &c) {
    return vec3(std::min(std::max(c.x(), 0.0), 1.0), std::min(std::max(c.y(), 0.0), 1.0),
                std::min(std::max(c.z(), 0.0), 1.0));
}

struct mtl_entry {
    vec3 Kd{0.8, 0.8, 0.8};
    vec3 Ks{0, 0, 0};
    double Ns = -1; // specular exponent, -1 = absent
    std::string map_Kd;
    std::string map_Bump; // normal / bump map (M73)
};

inline void load_mtl(const std::string &path, std::map<std::string, mtl_entry> &out) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "mtl missing: " << path << " (fallback material)\n";
        return;
    }
    std::string cur;
    std::map<std::string, bool> warned;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag) || tag[0] == '#')
            continue;
        if (tag == "newmtl") {
            if (ls >> cur)
                out[cur] = mtl_entry{};
        } else if (tag == "Kd" && !cur.empty()) {
            double r, g, b;
            if (ls >> r >> g >> b)
                out[cur].Kd = clamp01(vec3(r, g, b));
        } else if (tag == "Ks" && !cur.empty()) {
            double r, g, b;
            if (ls >> r >> g >> b)
                out[cur].Ks = clamp01(vec3(r, g, b));
        } else if (tag == "map_Kd" && !cur.empty()) {
            std::string p;
            if (ls >> p)
                out[cur].map_Kd = p;
        } else if ((tag == "map_Bump" || tag == "bump" || tag == "norm") && !cur.empty()) {
            std::string p;
            if (ls >> p)
                out[cur].map_Bump = p;
        } else if (tag == "Ns" && !cur.empty()) {
            double n;
            if (ls >> n)
                out[cur].Ns = n < 0 ? 0 : n;
        } else if (!warned[tag]) {
            warned[tag] = true;
            std::cerr << "mtl ignores '" << tag << "' in " << path << "\n";
        }
    }
}

// Ks present -> metal (Ns maps to GGX roughness, absent Ns stays mirror);
// else lambertian (map_Kd image wins, Ns unused there and warned).
// Ns->roughness follows Walter: r = sqrt(2/(Ns+2)) (Ns 0 -> 1, 1000 -> .04).
inline double mtl_roughness(double Ns) { return std::sqrt(2.0 / (Ns + 2.0)); }

inline std::shared_ptr<material> make_mtl_material(const mtl_entry &e,
                                                   const std::string &mtl_dir) {
    std::shared_ptr<material> mat;
    if (!e.map_Kd.empty()) {
        ppm_io::image img;
        if (stb_loader::load_image(mtl_dir + e.map_Kd, img) && !img.px.empty())
            mat = std::make_shared<lambertian>(
                std::make_shared<image_texture>(img.w, img.h, img.px));
        else
            std::cerr << "mtl map missing: " << mtl_dir + e.map_Kd << " (Kd fallback)\n";
    }
    if (!mat) {
        if (e.Ks.length_squared() > 0)
            mat = std::make_shared<metal>(e.Ks, e.Ns >= 0 ? mtl_roughness(e.Ns) : 0.0);
        else
            mat = std::make_shared<lambertian>(e.Kd);
    }
    if (!e.map_Bump.empty()) {
        ppm_io::image bimg;
        if (stb_loader::load_image(mtl_dir + e.map_Bump, bimg) && !bimg.px.empty()) {
            mat->set_normal_map(std::make_shared<image_texture>(bimg.w, bimg.h, bimg.px));
        } else {
            std::cerr << "mtl bump map missing: " << mtl_dir + e.map_Bump << "\n";
        }
    }
    return mat;
}

inline int fix_index(int idx, size_t n) {
    if (idx > 0)
        return idx - 1;
    return (int)n + idx; // negative = relative to end
}

// Corner reference: position + optional normal/uv indices (-1 = absent).
struct corner {
    int v = -1;
    int n = -1;
    int t = -1;
};

inline bool parse_corner(const std::string &tok, size_t nv, size_t nn, size_t nt,
                         corner &c) {
    // Forms: v | v/vt | v//vn | v/vt/vn. Empty fields skipped.
    size_t s1 = tok.find('/');
    std::string vs = (s1 == std::string::npos) ? tok : tok.substr(0, s1);
    if (vs.empty())
        return false;
    c.v = fix_index(std::stoi(vs), nv);
    if (s1 != std::string::npos) {
        size_t s2 = tok.find('/', s1 + 1);
        std::string ts = (s2 == std::string::npos) ? tok.substr(s1 + 1)
                                                   : tok.substr(s1 + 1, s2 - s1 - 1);
        if (!ts.empty())
            c.t = fix_index(std::stoi(ts), nt);
        if (s2 != std::string::npos) {
            std::string ns = tok.substr(s2 + 1);
            if (!ns.empty())
                c.n = fix_index(std::stoi(ns), nn);
        }
    }
    return c.v >= 0;
}

inline bool load_obj(const std::string &path, std::vector<std::shared_ptr<triangle>> &tris,
                     std::shared_ptr<material> mat) {
    std::ifstream f(path);
    if (!f)
        return false;
    std::vector<vec3> verts, normals;
    std::vector<std::pair<double, double>> uvs;
    std::map<std::string, mtl_entry> mtl;
    std::map<std::string, std::shared_ptr<material>> mtl_cache;
    std::string mtl_dir = file_dir(path);
    std::shared_ptr<material> cur_mtl; // null = caller fallback
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag) || tag[0] == '#')
            continue;
        if (tag == "mtllib") {
            std::string lib;
            if (ls >> lib)
                load_mtl(mtl_dir + lib, mtl);
        } else if (tag == "usemtl") {
            std::string name;
            if (!(ls >> name) || !mtl.count(name)) {
                std::cerr << "mtl unknown usemtl '" << name << "' (fallback)\n";
                cur_mtl.reset();
            } else {
                auto it = mtl_cache.find(name);
                if (it == mtl_cache.end())
                    it = mtl_cache
                             .emplace(name, make_mtl_material(mtl[name], mtl_dir))
                             .first;
                cur_mtl = it->second;
            }
        } else if (tag == "v") {
            double x, y, z;
            if (ls >> x >> y >> z)
                verts.emplace_back(x, y, z);
        } else if (tag == "vn") {
            double x, y, z;
            if (ls >> x >> y >> z)
                normals.emplace_back(x, y, z);
        } else if (tag == "vt") {
            double u, v;
            if (ls >> u >> v)
                uvs.emplace_back(u, v);
        } else if (tag == "f") {
            std::vector<corner> idx;
            std::string tok;
            while (ls >> tok) {
                corner c;
                if (parse_corner(tok, verts.size(), normals.size(), uvs.size(), c))
                    idx.push_back(c);
            }
            if (idx.size() < 3)
                continue;
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                const corner &a = idx[0], &b = idx[i], &cc = idx[i + 1];
                if ((size_t)a.v >= verts.size() || (size_t)b.v >= verts.size() ||
                    (size_t)cc.v >= verts.size())
                    continue;
                auto ok_n = [&](const corner &c) {
                    return c.n >= 0 && (size_t)c.n < normals.size();
                };
                auto ok_t = [&](const corner &c) {
                    return c.t >= 0 && (size_t)c.t < uvs.size();
                };
                bool smooth = ok_n(a) && ok_n(b) && ok_n(cc);
                bool textured = ok_t(a) && ok_t(b) && ok_t(cc);
                const vec3 &pa = verts[(size_t)a.v], &pb = verts[(size_t)b.v],
                           &pc = verts[(size_t)cc.v];
                if (smooth && textured)
                    tris.push_back(std::make_shared<triangle>(
                        pa, pb, pc, unit_vector(normals[(size_t)a.n]),
                        unit_vector(normals[(size_t)b.n]), unit_vector(normals[(size_t)cc.n]),
                        uvs[(size_t)a.t].first, uvs[(size_t)a.t].second,
                        uvs[(size_t)b.t].first, uvs[(size_t)b.t].second,
                        uvs[(size_t)cc.t].first, uvs[(size_t)cc.t].second, cur_mtl ? cur_mtl : mat));
                else if (smooth)
                    tris.push_back(std::make_shared<triangle>(
                        pa, pb, pc, unit_vector(normals[(size_t)a.n]),
                        unit_vector(normals[(size_t)b.n]),
                        unit_vector(normals[(size_t)cc.n]), cur_mtl ? cur_mtl : mat));
                else if (textured)
                    tris.push_back(std::make_shared<triangle>(
                        pa, pb, pc, uvs[(size_t)a.t].first, uvs[(size_t)a.t].second,
                        uvs[(size_t)b.t].first, uvs[(size_t)b.t].second,
                        uvs[(size_t)cc.t].first, uvs[(size_t)cc.t].second, cur_mtl ? cur_mtl : mat));
                else
                    tris.push_back(std::make_shared<triangle>(pa, pb, pc, cur_mtl ? cur_mtl : mat));
            }
        }
    }
    return !tris.empty();
}

} // namespace obj_loader
