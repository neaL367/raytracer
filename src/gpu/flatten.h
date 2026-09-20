#pragma once
// BVH flatten: CPU tree -> node + leaf-ref arrays for GPU upload.
// Pure logic, zero Vulkan: testable via unit tests. Pointer identity
// maps scene objects to typed-array slots (BVH sorts ptr copies).
// Order: data structs, image export, detail fns, entry point.
#include "host_scene.h"
#include "../accel/bvh.h"
#include "../scene/scene.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>

struct GPUNode {
    float bmin[4], bmax[4];
    int left = -1, right = -1, start = 0, count = 0;
};

struct GPURef {
    int type = -1; // 0 sphere, 1 quad, 2 tri
    int index = -1;
};

struct GPUImage {
    int w = 0, h = 0;
    const void *src = nullptr; // texture identity for dedupe
    std::vector<float> rgba; // top-first rows, linear HDR
};

struct flat_scene {
    gpu_scene gs;
    std::vector<GPUNode> nodes;
    std::vector<GPURef> refs;
    int nlights = 0; // leading emissive quads (NEE indexes [0, nlights))
    std::vector<GPUImage> images; // deduped by texture pointer
};

// Image-backed lambertian: registry assign, emit type-5 params.
// Same texture reuses its index (dedupe, not duplication).
inline bool try_image_export(flat_scene &out, const std::shared_ptr<lambertian> &lamb,
                             float alb[4], float alb2[4], float emit[4], float prm[4]) {
    auto it = std::dynamic_pointer_cast<image_texture>(lamb->tex_ref());
    if (!it)
        return false;
    int idx = -1;
    for (size_t k = 0; k < out.images.size(); ++k)
        if (out.images[k].src == it.get())
            idx = (int)k;
    if (idx < 0) {
        GPUImage gim;
        gim.w = it->width();
        gim.h = it->height();
        gim.src = it.get();
        gim.rgba.reserve((size_t)gim.w * gim.h * 4);
        for (const vec3 &p : it->texels()) {
            gim.rgba.push_back((float)p.x());
            gim.rgba.push_back((float)p.y());
            gim.rgba.push_back((float)p.z());
            gim.rgba.push_back(1.0f);
        }
        idx = (int)out.images.size();
        out.images.push_back(std::move(gim));
    }
    alb[0] = alb[1] = alb[2] = alb[3] = 0;
    alb2[0] = alb2[1] = alb2[2] = alb2[3] = 0;
    emit[0] = emit[1] = emit[2] = emit[3] = 0;
    prm[0] = 5;
    prm[1] = (float)idx;
    prm[2] = prm[3] = 0;
    return true;
}

namespace flat_detail {

inline void export_or_magenta(const std::shared_ptr<material> &m, float alb[4], float alb2[4],
                              float emit[4], float prm[4]) {
    if (m && m->export_gpu(alb, alb2, emit, prm))
        return;
    // Loud magenta fallback: visible, never silent.
    alb[0] = 1;
    alb[1] = 0;
    alb[2] = 1;
    alb2[0] = alb2[1] = alb2[2] = 0;
    emit[0] = emit[1] = emit[2] = 0;
    prm[0] = 0;
    prm[1] = prm[2] = prm[3] = 0;
}

// Material fill shared by all shapes: export, then image override.
inline void fill_material(flat_scene &out, const std::shared_ptr<material> &m, float alb[4],
                          float alb2[4], float emit[4], float prm[4]) {
    export_or_magenta(m, alb, alb2, emit, prm);
    if (auto l = std::dynamic_pointer_cast<lambertian>(m))
        try_image_export(out, l, alb, alb2, emit, prm);
}

inline bool push_prim(flat_scene &out, std::map<const hittable *, std::pair<int, int>> &id,
                      const std::shared_ptr<hittable> &o) {
    gpu_scene &gs = out.gs;
    if (auto s = std::dynamic_pointer_cast<sphere>(o)) {
        GPUSphere g{};
        vec3 c = s->center_ref();
        g.c[0] = (float)c.x();
        g.c[1] = (float)c.y();
        g.c[2] = (float)c.z();
        g.c[3] = (float)s->radius_val();
        fill_material(out, s->mat_ptr(), g.alb, g.alb2, g.emit, g.prm);
        id[o.get()] = {0, (int)gs.spheres.size()};
        gs.spheres.push_back(g);
        return true;
    }
    if (auto q = std::dynamic_pointer_cast<quad>(o)) {
        GPUQuad g{};
        vec3 Q = q->corner(), u = q->edge_u(), v = q->edge_v();
        g.Q[0] = (float)Q.x();
        g.Q[1] = (float)Q.y();
        g.Q[2] = (float)Q.z();
        g.u[0] = (float)u.x();
        g.u[1] = (float)u.y();
        g.u[2] = (float)u.z();
        g.v[0] = (float)v.x();
        g.v[1] = (float)v.y();
        g.v[2] = (float)v.z();
        fill_material(out, q->mat_ptr(), g.alb, g.alb2, g.emit, g.prm);
        id[o.get()] = {1, (int)gs.quads.size()};
        gs.quads.push_back(g);
        return true;
    }
    if (auto t = std::dynamic_pointer_cast<triangle>(o)) {
        GPUTri g{};
        for (int k = 0; k < 3; ++k) {
            vec3 vv = t->vert(k);
            vec3 nn = t->norm_vert(k);
            float *dst = (k == 0) ? g.a : ((k == 1) ? g.b : g.c);
            float *nd = (k == 0) ? g.n0 : ((k == 1) ? g.n1 : g.n2);
            dst[0] = (float)vv.x();
            dst[1] = (float)vv.y();
            dst[2] = (float)vv.z();
            nd[0] = (float)nn.x();
            nd[1] = (float)nn.y();
            nd[2] = (float)nn.z();
        }
        fill_material(out, t->mat_ptr(), g.alb, g.alb2, g.emit, g.prm);
        id[o.get()] = {2, (int)gs.tris.size()};
        gs.tris.push_back(g);
        return true;
    }
    return false; // unknown shape: fail loudly in flatten_scene
}

inline int flatten_node(const bvh_node &n, std::vector<GPUNode> &nodes,
                        std::vector<GPURef> &refs,
                        const std::map<const hittable *, std::pair<int, int>> &id) {
    int idx = (int)nodes.size();
    nodes.push_back(GPUNode{});
    aabb box = n.node_box();
    nodes[idx].bmin[0] = (float)box.minimum.x();
    nodes[idx].bmin[1] = (float)box.minimum.y();
    nodes[idx].bmin[2] = (float)box.minimum.z();
    nodes[idx].bmax[0] = (float)box.maximum.x();
    nodes[idx].bmax[1] = (float)box.maximum.y();
    nodes[idx].bmax[2] = (float)box.maximum.z();
    if (n.is_leaf()) {
        nodes[idx].start = (int)refs.size();
        nodes[idx].count = 0;
        for (const auto &p : n.leaf_prims()) {
            auto it = id.find(p.get());
            if (it == id.end())
                continue; // unmapped (should not happen): skip loudly below
            refs.push_back(GPURef{it->second.first, it->second.second});
            nodes[idx].count++;
        }
        return idx;
    }
    int l = flatten_node(*n.child(false), nodes, refs, id);
    int r = flatten_node(*n.child(true), nodes, refs, id);
    nodes[idx].left = l;
    nodes[idx].right = r;
    return idx;
}

} // namespace flat_detail

// Build typed arrays + SAH tree + flat nodes from a scene. False on
// unknown shapes only; lights stay quads in the same arrays.
// Emissive quads sort first so NEE indexing stays a prefix.
inline bool flatten_scene(const scene_data &scene, flat_scene &out) {
    out = flat_scene{};
    // Reject unknown shapes up front (image textures unlimited now).
    for (const auto &o : scene.objs)
        if (!std::dynamic_pointer_cast<sphere>(o) && !std::dynamic_pointer_cast<quad>(o) &&
            !std::dynamic_pointer_cast<triangle>(o))
            return false;
    std::vector<std::shared_ptr<hittable>> ordered = scene.objs;
    std::stable_partition(
        ordered.begin(), ordered.end(), [](const std::shared_ptr<hittable> &o) {
            auto q = std::dynamic_pointer_cast<quad>(o);
            if (!q)
                return false;
            float alb[4] = {}, alb2[4] = {}, emit[4] = {}, prm[4] = {};
            if (!q->mat_ptr() || !q->mat_ptr()->export_gpu(alb, alb2, emit, prm))
                return false;
            return prm[0] == 3;
        });
    std::map<const hittable *, std::pair<int, int>> id;
    for (const auto &o : ordered)
        if (!flat_detail::push_prim(out, id, o))
            return false;
    for (const auto &q : out.gs.quads)
        if (q.prm[0] == 3)
            out.nlights++;
    std::vector<std::shared_ptr<hittable>> objs = ordered; // ptr copies
    bvh_node root(objs, 0, objs.size(), true);
    flat_detail::flatten_node(root, out.nodes, out.refs, id);
    return true;
}
