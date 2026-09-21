#pragma once
// Flat 4-wide BVH for GPU upload: same 2+2 collapse rule as qbvh_node
// (children-of-children in DFS order), but stored as flat node + ref runs
// instead of recursive objects. The binary tree stays the CPU collapse
// source; this is its upload twin. CPU traversal mirror (flat_qbvh_hit)
// documents the GLSL order and is bit-exact vs bvh_node by construction.
#include "../geometry/hittable.h"
#include "bvh.h"

#include <cstddef>
#include <memory>
#include <vector>

struct flat_qslot {
    aabb box;
    bool leaf = true;
    int node = -1; // inner only: index into flat nodes
    std::vector<std::shared_ptr<hittable>> prims; // leaf only
};

struct flat_qnode {
    flat_qslot slot[4];
    int nslots = 0;
};

namespace flat_qbvh_detail {

inline void add_leaf_slot(flat_qnode &qn, const std::vector<std::shared_ptr<hittable>> &prims,
                          const aabb &box) {
    flat_qslot &s = qn.slot[qn.nslots++];
    s.leaf = true;
    s.prims = prims;
    s.box = box;
}

// Forward: inner slots need the node index before the child is built, so
// reserve the slot, build the child, then backfill.
inline int build_node(const bvh_node &b, std::vector<flat_qnode> &nodes);

inline void build_inner_slot(const bvh_node &b, std::vector<flat_qnode> &nodes,
                             int parent_idx) {
    // Build the child first (may reallocate), then backfill the reserved slot.
    int slot = nodes[(size_t)parent_idx].nslots++;
    int child_idx = build_node(b, nodes);
    flat_qslot &s = nodes[(size_t)parent_idx].slot[slot];
    s.leaf = false;
    s.node = child_idx;
    s.box = b.node_box();
}

inline int build_node(const bvh_node &b, std::vector<flat_qnode> &nodes) {
    // NOTE: never hold a node reference across recursion (emplace_back may
    // reallocate); always re-index by idx after each recursive build.
    int idx = (int)nodes.size();
    nodes.emplace_back();
    if (b.is_leaf()) {
        flat_qnode &qn = nodes[(size_t)idx];
        add_leaf_slot(qn, b.leaf_prims(), b.node_box());
        return idx;
    }
    const bvh_node *lr[2] = {b.child(false).get(), b.child(true).get()};
    for (int k = 0; k < 2; ++k) {
        if (!lr[k])
            continue;
        if (lr[k]->is_leaf()) {
            add_leaf_slot(nodes[(size_t)idx], lr[k]->leaf_prims(), lr[k]->node_box());
        } else {
            const bvh_node *l = lr[k]->child(false).get();
            const bvh_node *r = lr[k]->child(true).get();
            if (l) {
                build_inner_slot(*l, nodes, idx);
            }
            if (r) {
                build_inner_slot(*r, nodes, idx);
            }
        }
    }
    return idx;
}

} // namespace flat_qbvh_detail

// Entry: collapse a binary tree root into flat DFS nodes (index 0 = root).
inline void build_flat_qbvh(const bvh_node &root, std::vector<flat_qnode> &nodes) {
    nodes.clear();
    flat_qbvh_detail::build_node(root, nodes);
}

// CPU traversal mirror of the GLSL QBVH walk: mask once per node over
// (t_min, t_max), then scan slots low-to-high with closest narrowing —
// the same order qbvh_node::hit uses.
inline bool flat_qbvh_hit(const std::vector<flat_qnode> &nodes, const ray &r,
                          double t_min, double t_max, hit_record &rec) {
    if (nodes.empty())
        return false;
    int stack[32];
    int sp = 0;
    stack[sp++] = 0;
    hit_record tmp;
    bool any = false;
    double closest = t_max;
    while (sp > 0) {
        const flat_qnode &qn = nodes[(size_t)stack[--sp]];
        unsigned mask = 0;
        for (int s = 0; s < qn.nslots; ++s)
            if (qn.slot[s].box.hit(r, t_min, t_max))
                mask |= 1u << (unsigned)s;
        // Push high-to-low so slot 0 pops first (DFS order).
        for (int s = qn.nslots - 1; s >= 0; --s) {
            if (!(mask & (1u << (unsigned)s)))
                continue;
            const flat_qslot &sl = qn.slot[s];
            if (sl.leaf) {
                for (const auto &p : sl.prims) {
                    if (p->hit(r, t_min, closest, tmp)) {
                        closest = tmp.t;
                        rec = tmp;
                        any = true;
                    }
                }
            } else if (sp < 32) {
                stack[sp++] = sl.node;
            }
        }
    }
    return any;
}

// std430 twin of the GLSL GPUQNode (vec4[4] + ivec4 x3 = 176 bytes).
struct GPUQNode {
    float bmin[4][4], bmax[4][4];
    int child[4], start[4], count[4];
};

inline GPUQNode to_gpu_qnode(const flat_qnode &qn) {
    GPUQNode g{};
    for (int s = 0; s < 4; ++s) {
        aabb b;
        bool live = s < qn.nslots;
        if (live)
            b = qn.slot[s].box;
        else
            b = aabb(vec3(1e30, 1e30, 1e30), vec3(-1e30, -1e30, -1e30));
        g.bmin[s][0] = (float)b.minimum.x();
        g.bmin[s][1] = (float)b.minimum.y();
        g.bmin[s][2] = (float)b.minimum.z();
        g.bmin[s][3] = 0;
        g.bmax[s][0] = (float)b.maximum.x();
        g.bmax[s][1] = (float)b.maximum.y();
        g.bmax[s][2] = (float)b.maximum.z();
        g.bmax[s][3] = 0;
        g.child[s] = (live && !qn.slot[s].leaf) ? qn.slot[s].node : -1;
        g.start[s] = 0;
        g.count[s] = 0;
    }
    return g;
}
