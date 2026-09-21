#pragma once
#include "../geometry/hittable.h"
#include "../core/bench_stats.h"
#include "bvh.h"

#include <algorithm>
#include <memory>
#include <vector>

#if defined(_M_X64) || defined(_M_AMD64) || defined(__SSE2__) || defined(__x86_64__)
#define QBVH_SSE2 1
#include <emmintrin.h>
#endif

// 4-wide BVH over the binary SAH tree: collapse children-of-children
// (2+2 slots, never overflows); leaves stay binary leaves 1:1.
// DFS slot order + strict-less inter-slot + last-wins intra-leaf mirrors
// bvh_node tie rules exactly -> bit-identical renders by construction.
// One 4-wide SSE2 slab test per node (NaN-safe like aabb::hit); leaves
// stay scalar. GPU flatten keeps the binary tree (untouched).
class qbvh_node : public hittable {
public:
    qbvh_node() {}
    qbvh_node(std::vector<std::shared_ptr<hittable>> &objs, size_t start, size_t end,
              bool use_sah = true) {
        bvh_node tmp(objs, start, end, use_sah);
        box = tmp.node_box();
        collapse(tmp);
    }
    // Collapse an existing binary tree (same DFS order -> bit-exact twin).
    explicit qbvh_node(const bvh_node &b) {
        box = b.node_box();
        collapse(b);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        count_box();
        count_box();
        count_box();
        count_box();
        unsigned mask = box_mask(r, t_min, t_max);
        if (!mask)
            return false;
        hit_record tmp;
        bool any = false;
        double closest = t_max;
        for (int s = 0; s < nkids; ++s) {
            if (!(mask & (1u << (unsigned)s)))
                continue;
            const child &c = kids[(size_t)s];
            if (c.leaf) {
                for (const auto &p : c.prims) {
                    ::count_prim();
                    if (p->hit(r, t_min, closest, tmp)) {
                        closest = tmp.t;
                        rec = tmp;
                        any = true;
                    }
                }
            } else if (c.node->hit(r, t_min, closest, tmp) && tmp.t < closest) {
                closest = tmp.t;
                rec = tmp;
                any = true;
            }
        }
        return any;
    }

    bool hit_any(const ray &r, double t_min, double t_max) const override {
        ::count_box();
        ::count_box();
        ::count_box();
        ::count_box();
        unsigned mask = box_mask(r, t_min, t_max);
        if (!mask)
            return false;
        for (int s = 0; s < nkids; ++s) {
            if (!(mask & (1u << (unsigned)s)))
                continue;
            const child &c = kids[(size_t)s];
            if (c.leaf) {
                for (const auto &p : c.prims) {
                    ::count_prim();
                    if (p->hit_any(r, t_min, t_max))
                        return true;
                }
            } else if (c.node->hit_any(r, t_min, t_max)) {
                return true;
            }
        }
        return false;
    }

    bool bounding_box(aabb &b) const override {
        b = box;
        return true;
    }

private:
    struct child {
        aabb box;
        bool leaf = true;
        std::vector<std::shared_ptr<hittable>> prims; // leaf only
        std::unique_ptr<qbvh_node> node; // inner only
    };

    aabb box;
    child kids[4];
    int nkids = 0;
    // SoA slabs as doubles: the SSE test runs fp64 lane math identical to
    // scalar aabb::hit (same ops, same order) -> bit-exact traversal.
    // Empty slots inverted -> never hit.
    double mnx[4], mny[4], mnz[4], mxx[4], mxy[4], mxz[4];

    void add_leaf(const std::vector<std::shared_ptr<hittable>> &prims, const aabb &b) {
        child &c = kids[(size_t)nkids++];
        c.leaf = true;
        c.prims = prims;
        c.box = b;
        set_slab(nkids - 1, b);
    }

    void add_inner(const bvh_node &b) {
        child &c = kids[(size_t)nkids++];
        c.leaf = false;
        c.node = std::unique_ptr<qbvh_node>(new qbvh_node(b));
        c.box = b.node_box();
        set_slab(nkids - 1, c.box);
    }

    void set_slab(int s, const aabb &b) {
        mnx[s] = b.minimum.x();
        mny[s] = b.minimum.y();
        mnz[s] = b.minimum.z();
        mxx[s] = b.maximum.x();
        mxy[s] = b.maximum.y();
        mxz[s] = b.maximum.z();
    }

    // Slots from children-of-children in DFS order (2+2 cap).
    void collapse(const bvh_node &b) {
        for (int i = 0; i < 4; ++i) {
            mnx[i] = 1e30;
            mny[i] = 1e30;
            mnz[i] = 1e30;
            mxx[i] = -1e30;
            mxy[i] = -1e30;
            mxz[i] = -1e30;
        }
        if (b.is_leaf()) {
            add_leaf(b.leaf_prims(), b.node_box());
            return;
        }
        const bvh_node *lr[2] = {b.child(false).get(), b.child(true).get()};
        for (int k = 0; k < 2; ++k) {
            if (!lr[k])
                continue;
            if (lr[k]->is_leaf()) {
                add_leaf(lr[k]->leaf_prims(), lr[k]->node_box());
            } else {
                const bvh_node *l = lr[k]->child(false).get();
                const bvh_node *r = lr[k]->child(true).get();
                if (l)
                    add_inner(*l);
                if (r)
                    add_inner(*r);
            }
        }
    }

    // 4-wide slab test in fp64 lane math, identical to scalar aabb::hit:
    // per-axis scalar invD sign swap, compare+blend narrowing (NaN keeps
    // the old bound, like scalar), strict miss. Two __m128d passes.
    // (_mm_blendv avoided: SSE4.1; and/or + andnot is SSE2.)
    unsigned box_mask(const ray &r, double t_min, double t_max) const {
#ifdef QBVH_SSE2
        const vec3 &o = r.origin();
        const vec3 &d = r.direction();
        double ox[3] = {o.x(), o.y(), o.z()};
        double dd[3] = {d.x(), d.y(), d.z()};
        const double *mns[3] = {mnx, mny, mnz};
        const double *mxs[3] = {mxx, mxy, mxz};
        unsigned mask = 0;
        for (int half = 0; half < 2; ++half) {
            __m128d tmn = _mm_set1_pd(t_min);
            __m128d tmx = _mm_set1_pd(t_max);
            for (int a = 0; a < 3; ++a) {
                double invD = 1.0 / dd[a];
                __m128d lo = _mm_loadu_pd(mns[a] + half * 2);
                __m128d hi = _mm_loadu_pd(mxs[a] + half * 2);
                if (invD < 0.0) {
                    __m128d t = lo;
                    lo = hi;
                    hi = t;
                }
                __m128d o4 = _mm_set1_pd(ox[a]);
                __m128d iv = _mm_set1_pd(invD);
                __m128d t0 = _mm_mul_pd(_mm_sub_pd(lo, o4), iv);
                __m128d t1 = _mm_mul_pd(_mm_sub_pd(hi, o4), iv);
                __m128d gt0 = _mm_cmpgt_pd(t0, tmn);
                tmn = _mm_or_pd(_mm_and_pd(gt0, t0), _mm_andnot_pd(gt0, tmn));
                __m128d lt1 = _mm_cmplt_pd(t1, tmx);
                tmx = _mm_or_pd(_mm_and_pd(lt1, t1), _mm_andnot_pd(lt1, tmx));
            }
            mask |= (unsigned)_mm_movemask_pd(_mm_cmpgt_pd(tmx, tmn)) << (half * 2);
        }
        return mask & 0xF;
#else
        unsigned mask = 0;
        for (int s = 0; s < nkids; ++s) {
            aabb b(vec3(mnx[s], mny[s], mnz[s]), vec3(mxx[s], mxy[s], mxz[s]));
            if (b.hit(r, t_min, t_max))
                mask |= 1u << (unsigned)s;
        }
        return mask;
#endif
    }
};
