#pragma once
// BVH flatten: CPU tree -> node + leaf-ref arrays for GPU upload.
// Pure logic, zero Vulkan: testable via unit tests. Pointer identity
// maps scene objects to typed-array slots (BVH sorts ptr copies).
// Order: data structs, image export, detail fns, entry point.
#include "host_scene.h"
#include "../accel/bvh.h"
#include "../accel/qbvh.h"
#include "../accel/qbvh_flat.h"
#include "../geometry/instance.h"
#include "../geometry/disk.h"
#include "../geometry/cylinder.h"
#include "../scene/scene.h"

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>

struct GPURef {
    int type = -1; // 0 sphere, 1 quad, 2 tri
    int index = -1;
};
static_assert(sizeof(GPURef) == 8, "GPURef must be 8 bytes to match GLSL ivec2 layout");

struct GPUImage {
    int w = 0, h = 0;
    int levels = 1; // mipmap count; blob holds L0..Ln consecutively
    float span = 8.0f; // world units one full texture spans (LOD)
    const void *src = nullptr; // texture identity for dedupe
    std::vector<float> rgba; // top-first rows, linear HDR
};

struct flat_scene {
    gpu_scene gs;
    std::vector<GPUQNode> nodes; // 4-wide QBVH, index 0 = root
    std::vector<GPURef> refs; // leaf runs, one (start,count) per leaf slot
    int nlights = 0; // NEE light count (entries in light_table)
    std::vector<std::pair<int, int>> light_table; // (type, index): 0 quad, 1 sphere, 2 tri
    std::vector<GPUImage> images; // deduped by texture pointer
};

// Image registry: assign an index, upload the mip pyramid once, and reuse
// the same index for repeated textures (dedupe, not duplication).
inline int register_image(flat_scene &out, const std::shared_ptr<image_texture> &it) {
    for (size_t k = 0; k < out.images.size(); ++k)
        if (out.images[k].src == it.get())
            return (int)k;
    GPUImage gim;
    gim.w = it->width();
    gim.h = it->height();
    gim.src = it.get();
    gim.span = (float)it->span();
    gim.levels = (int)it->mip_chain().size();
    for (const auto &lv : it->mip_chain()) {
        gim.rgba.reserve(gim.rgba.size() + (size_t)lv.w * lv.h * 4);
        for (const vec3 &p : lv.px) {
            gim.rgba.push_back((float)p.x());
            gim.rgba.push_back((float)p.y());
            gim.rgba.push_back((float)p.z());
            gim.rgba.push_back(1.0f);
        }
    }
    int idx = (int)out.images.size();
    out.images.push_back(std::move(gim));
    return idx;
}

// Image-backed lambertian: registry assign, emit type-5 params.
inline bool try_image_export(flat_scene &out, const std::shared_ptr<lambertian> &lamb,
                             float alb[4], float alb2[4], float emit[4], float prm[4]) {
    auto it = std::dynamic_pointer_cast<image_texture>(lamb->tex_ref());
    if (!it)
        return false;
    int idx = register_image(out, it);
    alb[0] = alb[1] = alb[2] = alb[3] = 0;
    alb2[0] = alb2[1] = alb2[2] = alb2[3] = 0;
    emit[0] = emit[1] = emit[2] = emit[3] = 0;
    prm[0] = static_cast<float>(MatType::IMAGE);
    prm[1] = (float)idx;
    prm[2] = prm[3] = 0;
    return true;
}

// Image-backed emitter: same registry, but stays MatType::EMIT. params.y
// carries the image index plus one (0 keeps the legacy flat emitter,
// avoiding an ambiguous index-0 encoding).
inline bool try_emissive_image_export(flat_scene &out,
                                      const std::shared_ptr<diffuse_light> &emit_mat,
                                      float alb[4], float alb2[4], float emit[4],
                                      float prm[4]) {
    auto it = std::dynamic_pointer_cast<image_texture>(emit_mat->tex_ref());
    if (!it)
        return false;
    int idx = register_image(out, it);
    alb[0] = alb[1] = alb[2] = alb[3] = 0;
    alb2[0] = alb2[1] = alb2[2] = alb2[3] = 0;
    emit[0] = emit[1] = emit[2] = emit[3] = 0;
    prm[0] = static_cast<float>(MatType::EMIT);
    prm[1] = (float)(idx + 1);
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
    prm[0] = static_cast<float>(MatType::SOLID);
    prm[1] = prm[2] = prm[3] = 0;
}

// Material fill shared by all shapes: export, then image override.
inline void fill_material(flat_scene &out, const std::shared_ptr<material> &m, float alb[4],
                          float alb2[4], float emit[4], float prm[4]) {
    export_or_magenta(m, alb, alb2, emit, prm);
    if (auto l = std::dynamic_pointer_cast<lambertian>(m))
        try_image_export(out, l, alb, alb2, emit, prm);
    if (auto e = std::dynamic_pointer_cast<diffuse_light>(m))
        try_emissive_image_export(out, e, alb, alb2, emit, prm);
}

inline bool push_prim(flat_scene &out, std::map<const hittable *, std::pair<int, int>> &id,
                       const std::shared_ptr<hittable> &o) {
    gpu_scene &gs = out.gs;
    // Fog volumes upload as type-6 sphere slots (boundary + density).
    // Non-sphere borders fail loudly (demo uses spheres).
    if (auto m = std::dynamic_pointer_cast<constant_medium>(o)) {
        auto b = std::dynamic_pointer_cast<sphere>(m->border_ref());
        if (!b)
            return false;
        GPUSphere g{};
        vec3 c = b->center_ref();
        g.c[0] = (float)c.x();
        g.c[1] = (float)c.y();
        g.c[2] = (float)c.z();
        g.c[3] = (float)b->radius_val();
        hit_record dummy;
        vec3 alb = m->phase_ref()->surface_albedo(dummy);
        g.alb[0] = (float)alb.x();
        g.alb[1] = (float)alb.y();
        g.alb[2] = (float)alb.z();
        g.prm[0] = static_cast<float>(MatType::FOG);
        g.prm[1] = (float)m->density_val();
        id[o.get()] = {0, (int)gs.spheres.size()};
        gs.spheres.push_back(g);
        return true;
    }
    // Heterogeneous twin: type-8 slot, modulation freqs ride alb2.
    if (auto hm = std::dynamic_pointer_cast<heterogeneous_medium>(o)) {
        auto b = std::dynamic_pointer_cast<sphere>(hm->border_ref());
        if (!b)
            return false;
        GPUSphere g{};
        vec3 c = b->center_ref();
        g.c[0] = (float)c.x();
        g.c[1] = (float)c.y();
        g.c[2] = (float)c.z();
        g.c[3] = (float)b->radius_val();
        hit_record dummy;
        vec3 alb = hm->phase_ref()->surface_albedo(dummy);
        g.alb[0] = (float)alb.x();
        g.alb[1] = (float)alb.y();
        g.alb[2] = (float)alb.z();
        vec3 fr = hm->freqs();
        g.alb2[0] = (float)fr.x();
        g.alb2[1] = (float)fr.y();
        g.alb2[2] = (float)fr.z();
        g.prm[0] = static_cast<float>(MatType::HET);
        g.prm[1] = (float)hm->density_val();
        id[o.get()] = {0, (int)gs.spheres.size()};
        gs.spheres.push_back(g);
        return true;
    }
    if (auto s = std::dynamic_pointer_cast<sphere>(o)) {
        GPUSphere g{};
        vec3 c = s->center_ref();
        g.c[0] = (float)c.x();
        g.c[1] = (float)c.y();
        g.c[2] = (float)c.z();
        g.c[3] = (float)s->radius_val();
        vec3 c1 = s->center1_ref();
        g.c1[0] = (float)c1.x();
        g.c1[1] = (float)c1.y();
        g.c1[2] = (float)c1.z();
        double t0 = 0, t1 = 1;
        s->time_range(t0, t1);
        g.tm[0] = (float)t0;
        g.tm[1] = (float)t1;
        fill_material(out, s->mat_ptr(), g.alb, g.alb2, g.emit, g.prm);
        // Motion flag: centers differ (static spheres keep c0, flag 0).
        if (c.x() != c1.x() || c.y() != c1.y() || c.z() != c1.z())
            g.prm[3] = 1;
        id[o.get()] = {0, (int)gs.spheres.size()};
        gs.spheres.push_back(g);
        if (g.prm[0] == static_cast<float>(MatType::EMIT) ||
            (g.prm[0] != static_cast<float>(MatType::FOG) &&
             g.prm[0] != static_cast<float>(MatType::DISNEY) &&
             g.emit[0] + g.emit[1] + g.emit[2] > 0))
            out.light_table.push_back({1, (int)gs.spheres.size() - 1});
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
        if (g.prm[0] == static_cast<float>(MatType::EMIT) ||
            (g.prm[0] != static_cast<float>(MatType::DISNEY) &&
             g.emit[0] + g.emit[1] + g.emit[2] > 0))
            out.light_table.push_back({0, (int)gs.quads.size() - 1});
        return true;
    }
    if (auto t = std::dynamic_pointer_cast<triangle>(o)) {
        GPUTri g{};
        for (int k = 0; k < 3; ++k) {
            vec3 vv = t->vert(k);
            vec3 nn = t->norm_vert(k);
            vec3 uv = t->uv_present() ? t->uv_vert(k) : vec3(0, 0, 0);
            float *dst = (k == 0) ? g.a : ((k == 1) ? g.b : g.c);
            float *nd = (k == 0) ? g.n0 : ((k == 1) ? g.n1 : g.n2);
            dst[0] = (float)vv.x();
            dst[1] = (float)vv.y();
            dst[2] = (float)vv.z();
            nd[0] = (float)nn.x();
            nd[1] = (float)nn.y();
            nd[2] = (float)nn.z();
            if (k < 2) {
                g.tuvA[k * 2] = (float)uv.x();
                g.tuvA[k * 2 + 1] = (float)uv.y();
            } else {
                g.tuvB[0] = (float)uv.x();
                g.tuvB[1] = (float)uv.y();
            }
        }
        fill_material(out, t->mat_ptr(), g.alb, g.alb2, g.emit, g.prm);
        if (t->uv_present())
            g.prm[3] = 1; // corner-UV blend on device
        for (int k = 0; k < 3; ++k) {
            vec3 v1 = t->vert1(k);
            float *m1 = (k == 0) ? g.a1 : ((k == 1) ? g.b1 : g.c1);
            m1[0] = (float)v1.x();
            m1[1] = (float)v1.y();
            m1[2] = (float)v1.z();
        }
        double t0 = 0, t1 = 1;
        t->time_range(t0, t1);
        g.tm[0] = (float)t0;
        g.tm[1] = (float)t1;
        id[o.get()] = {2, (int)gs.tris.size()};
        gs.tris.push_back(g);
        if (g.prm[0] == static_cast<float>(MatType::EMIT) ||
            (g.prm[0] != static_cast<float>(MatType::DISNEY) &&
             g.emit[0] + g.emit[1] + g.emit[2] > 0))
            out.light_table.push_back({2, (int)gs.tris.size() - 1});
        return true;
    }
    return false; // unknown shape: fail loudly in flatten_scene
}

// DFS flatten of one flat-QBVH node: pre-register the slot, then fill leaf
// runs (appended to refs) and recurse inner slots. Returns the output index.
inline int flatten_qnode_at(const std::vector<flat_qnode> &all, int idx,
                            std::vector<GPUQNode> &nodes, std::vector<GPURef> &refs,
                            const std::map<const hittable *, std::pair<int, int>> &id) {
    int out_idx = (int)nodes.size();
    nodes.push_back(GPUQNode{});
    const flat_qnode &qn = all[(size_t)idx];
    GPUQNode g = to_gpu_qnode(qn);
    for (int s = 0; s < qn.nslots; ++s) {
        if (qn.slot[s].leaf) {
            g.start[s] = (int)refs.size();
            g.count[s] = 0;
            for (const auto &p : qn.slot[s].prims) {
                auto it = id.find(p.get());
                if (it == id.end())
                    continue; // unmapped (should not happen): skip loudly below
                refs.push_back(GPURef{it->second.first, it->second.second});
                g.count[s]++;
            }
        } else {
            g.child[s] = flatten_qnode_at(all, qn.slot[s].node, nodes, refs, id);
        }
    }
    nodes[(size_t)out_idx] = g;
    return out_idx;
}

} // namespace flat_detail

// GPU instance/container expander: instances bake to world-space prims so the
// BVH + shaders only ever see plain prims (CPU keeps true instances for exact
// normals). Rotation-invariant spheres bake as transformed centers; quads bake
// as rotated Q/u/v (same math as instance_detail::collect_baked_quads).
// Containers (list, bvh_node, qbvh_node) expand to leaves. Volumes pass
// through only untransformed (transformed media unsupported: fail loudly).
inline bool expand_for_gpu(const std::shared_ptr<hittable> &o, double c, double s,
                           const vec3 &T, std::vector<std::shared_ptr<hittable>> &out,
                           std::vector<std::shared_ptr<quad>> &quad_owned,
                           std::vector<std::shared_ptr<sphere>> &sphere_owned,
                           std::vector<std::shared_ptr<triangle>> &tri_owned) {
    // Identity: pass the original through (pointer identity preserved for
    // the BVH id map; untransformed motion uploads natively).
    bool ident = (c == 1.0 && s == 0.0 && T.x() == 0.0 && T.y() == 0.0 && T.z() == 0.0);
    if (auto q = std::dynamic_pointer_cast<quad>(o)) {
        if (ident) {
            out.push_back(o);
            return true;
        }
        vec3 Qw = instance_detail::rot_point(q->corner(), c, s) + T;
        vec3 uw = instance_detail::rot_dir(q->edge_u(), c, s);
        vec3 vw = instance_detail::rot_dir(q->edge_v(), c, s);
        auto qp = std::make_shared<quad>(Qw, uw, vw, q->mat_ptr());
        quad_owned.push_back(qp);
        out.push_back(qp);
        return true;
    }
    if (auto sp = std::dynamic_pointer_cast<sphere>(o)) {
        if (ident) {
            out.push_back(o);
            return true;
        }
        // Rigid motion preserves linear motion: bake both endpoints + range.
        double t0 = 0, t1 = 1;
        sp->time_range(t0, t1);
        vec3 C0w = instance_detail::rot_point(sp->center_ref(), c, s) + T;
        vec3 C1w = instance_detail::rot_point(sp->center1_ref(), c, s) + T;
        auto np = std::make_shared<sphere>(C0w, C1w, t0, t1, sp->radius_val(), sp->mat_ptr());
        sphere_owned.push_back(np);
        out.push_back(np);
        return true;
    }
    if (auto list = std::dynamic_pointer_cast<hittable_list>(o)) {
        for (const auto &child : list->children())
            if (!expand_for_gpu(child, c, s, T, out, quad_owned, sphere_owned, tri_owned))
                return false;
        return true;
    }
    if (auto bn = std::dynamic_pointer_cast<bvh_node>(o)) {
        if (bn->is_leaf()) {
            for (const auto &p : bn->leaf_prims())
                if (!expand_for_gpu(p, c, s, T, out, quad_owned, sphere_owned, tri_owned))
                    return false;
            return true;
        }
        return expand_for_gpu(bn->child(false), c, s, T, out, quad_owned, sphere_owned, tri_owned) &&
               expand_for_gpu(bn->child(true), c, s, T, out, quad_owned, sphere_owned, tri_owned);
    }
    if (auto qn = std::dynamic_pointer_cast<qbvh_node>(o)) {
        std::vector<std::shared_ptr<hittable>> prims;
        qn->collect_prims(prims);
        for (const auto &p : prims)
            if (!expand_for_gpu(p, c, s, T, out, quad_owned, sphere_owned, tri_owned))
                return false;
        return true;
    }
    if (auto tr = std::dynamic_pointer_cast<triangle>(o)) {
        if (ident) {
            out.push_back(o); // mesh tris upload natively; instances unsupported
            return true;
        }
        (void)tr;
        return false; // transformed triangle: fail loudly (as before)
    }
    if (auto dsk = std::dynamic_pointer_cast<disk>(o)) {
        vec3 c_pos = instance_detail::rot_point(dsk->get_center(), c, s) + T;
        vec3 n_dir = unit_vector(instance_detail::rot_dir(dsk->get_normal(), c, s));
        vec3 up = (std::abs(n_dir.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 u_ax = unit_vector(cross(n_dir, up));
        vec3 v_ax = cross(n_dir, u_ax);
        int segs = 24;
        double r_out = dsk->get_radius();
        double r_in = dsk->get_inner_radius();
        double dtheta = 2.0 * 3.1415926535897932385 / (double)segs;
        for (int k = 0; k < segs; ++k) {
            double th0 = (double)k * dtheta;
            double th1 = th0 + dtheta;
            vec3 dir0 = std::cos(th0) * u_ax + std::sin(th0) * v_ax;
            vec3 dir1 = std::cos(th1) * u_ax + std::sin(th1) * v_ax;
            if (r_in <= 1e-6) {
                auto tri = std::make_shared<triangle>(c_pos, c_pos + r_out * dir0, c_pos + r_out * dir1, dsk->mat_ptr());
                tri_owned.push_back(tri);
                out.push_back(tri);
            } else {
                auto tri1 = std::make_shared<triangle>(c_pos + r_in * dir0, c_pos + r_out * dir0, c_pos + r_out * dir1, dsk->mat_ptr());
                auto tri2 = std::make_shared<triangle>(c_pos + r_in * dir0, c_pos + r_out * dir1, c_pos + r_in * dir1, dsk->mat_ptr());
                tri_owned.push_back(tri1);
                tri_owned.push_back(tri2);
                out.push_back(tri1);
                out.push_back(tri2);
            }
        }
        return true;
    }
    if (auto cyl = std::dynamic_pointer_cast<cylinder>(o)) {
        vec3 p0 = instance_detail::rot_point(cyl->get_base(), c, s) + T;
        vec3 p1 = instance_detail::rot_point(cyl->get_top(), c, s) + T;
        vec3 ax = p1 - p0;
        double len = ax.length();
        vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
        vec3 up = (std::abs(ax_u.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 u_ax = unit_vector(cross(ax_u, up));
        vec3 v_ax = cross(ax_u, u_ax);
        int segs = 24;
        double r = cyl->get_radius();
        double dtheta = 2.0 * 3.1415926535897932385 / (double)segs;
        for (int k = 0; k < segs; ++k) {
            double th0 = (double)k * dtheta;
            double th1 = th0 + dtheta;
            vec3 r0 = std::cos(th0) * u_ax + std::sin(th0) * v_ax;
            vec3 r1 = std::cos(th1) * u_ax + std::sin(th1) * v_ax;
            vec3 b0 = p0 + r * r0, b1 = p0 + r * r1;
            vec3 t0 = p1 + r * r0, t1 = p1 + r * r1;
            auto sw1 = std::make_shared<triangle>(b0, b1, t1, cyl->mat_ptr());
            auto sw2 = std::make_shared<triangle>(b0, t1, t0, cyl->mat_ptr());
            tri_owned.push_back(sw1);
            tri_owned.push_back(sw2);
            out.push_back(sw1);
            out.push_back(sw2);
            auto cap_b = std::make_shared<triangle>(p0, b1, b0, cyl->mat_ptr());
            auto cap_t = std::make_shared<triangle>(p1, t0, t1, cyl->mat_ptr());
            tri_owned.push_back(cap_b);
            tri_owned.push_back(cap_t);
            out.push_back(cap_b);
            out.push_back(cap_t);
        }
        return true;
    }
    if (auto tr2 = std::dynamic_pointer_cast<translate>(o)) {
        vec3 T2 = instance_detail::rot_point(tr2->offset(), c, s) + T;
        return expand_for_gpu(tr2->inner_ref(), c, s, T2, out, quad_owned, sphere_owned, tri_owned);
    }
    if (auto ry = std::dynamic_pointer_cast<rotate_y>(o)) {
        double ci = ry->cos_theta(), si = ry->sin_theta();
        return expand_for_gpu(ry->inner_ref(), c * ci - s * si, s * ci + c * si, T, out,
                              quad_owned, sphere_owned, tri_owned);
    }
    if ((std::dynamic_pointer_cast<constant_medium>(o) ||
         std::dynamic_pointer_cast<heterogeneous_medium>(o)) &&
        c == 1.0 && s == 0.0 && T.x() == 0.0 && T.y() == 0.0 && T.z() == 0.0) {
        out.push_back(o); // untransformed volume: whitelist below checks border
        return true;
    }
    return false; // unknown shape (or transformed volume): fail loudly
}

// Build typed arrays + SAH tree + flat nodes from a scene. False on
// unknown shapes only. Emissive prims (any shape) register in the light
// table; the emissive-first partition is legacy order, kept stable.
// Instances bake to world-space quads first, so the BVH + shaders only
// ever see plain prims (CPU keeps true instances for exact normals).
inline bool flatten_scene(const scene_data &scene, flat_scene &out) {
    out = flat_scene{};
    // Expand instances + containers up front; baked prims are owned here.
    std::vector<std::shared_ptr<hittable>> expanded;
    std::vector<std::shared_ptr<quad>> quad_owned;
    std::vector<std::shared_ptr<sphere>> sphere_owned;
    std::vector<std::shared_ptr<triangle>> tri_owned;
    for (const auto &o : scene.objs) {
        if (!expand_for_gpu(o, 1.0, 0.0, vec3(0, 0, 0), expanded, quad_owned,
                            sphere_owned, tri_owned))
            return false;
    }
    // Reject unknown shapes up front (image textures unlimited now).
    // Fog media allowed only over sphere borders (demo scope).
    for (const auto &o : expanded) {
        if (std::dynamic_pointer_cast<sphere>(o) || std::dynamic_pointer_cast<quad>(o) ||
            std::dynamic_pointer_cast<triangle>(o))
            continue;
        auto med = std::dynamic_pointer_cast<constant_medium>(o);
        if (med && std::dynamic_pointer_cast<sphere>(med->border_ref()))
            continue;
        auto het = std::dynamic_pointer_cast<heterogeneous_medium>(o);
        if (het && std::dynamic_pointer_cast<sphere>(het->border_ref()))
            continue;
        return false;
    }
    std::vector<std::shared_ptr<hittable>> ordered = expanded;
    std::stable_partition(
        ordered.begin(), ordered.end(), [](const std::shared_ptr<hittable> &o) {
            auto q = std::dynamic_pointer_cast<quad>(o);
            if (!q || !q->mat_ptr())
                return false;
            // Flat and image-backed emitters sort first; textured emission
            // exports through the image path even though export_gpu refuses.
            // Other emitter patterns stay in place and keep magenta fallback.
            if (auto dl = std::dynamic_pointer_cast<diffuse_light>(q->mat_ptr())) {
                float dalb[4] = {}, dalb2[4] = {}, demit[4] = {}, dprm[4] = {};
                if (dl->export_gpu(dalb, dalb2, demit, dprm))
                    return dprm[0] == static_cast<float>(MatType::EMIT);
                return (bool)std::dynamic_pointer_cast<image_texture>(dl->tex_ref());
            }
            float alb[4] = {}, alb2[4] = {}, emit[4] = {}, prm[4] = {};
            if (!q->mat_ptr()->export_gpu(alb, alb2, emit, prm))
                return false;
            return prm[0] == static_cast<float>(MatType::EMIT);
        });
    std::map<const hittable *, std::pair<int, int>> id;
    for (const auto &o : ordered)
        if (!flat_detail::push_prim(out, id, o))
            return false;
    out.nlights = (int)out.light_table.size();
    std::vector<std::shared_ptr<hittable>> objs = ordered; // ptr copies
    bvh_node root(objs, 0, objs.size(), true);
    // 4-wide collapse over the same SAH tree the CPU traverses, then flat
    // upload (one ref run per leaf slot; binary path deleted in M39).
    std::vector<flat_qnode> qtree;
    build_flat_qbvh(root, qtree);
    flat_detail::flatten_qnode_at(qtree, 0, out.nodes, out.refs, id);
    return true;
}
