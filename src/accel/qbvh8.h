#pragma once
#include "../geometry/hittable.h"
#include "../core/bench_stats.h"
#include "bvh.h"

#include <algorithm>
#include <memory>
#include <vector>

#if defined(__AVX2__) || defined(__AVX__)
#define QBVH8_AVX 1
#include <immintrin.h>
#endif

// 8-wide BVH over binary SAH tree: collapse children-of-children plus one
// grandchild level (up to 8 slots, DFS order, cap 8); leaves stay 1:1.
// Float SoA slabs, single 8-wide AVX slab test, scalar fallback otherwise.
// Precision: float slabs may differ in last ulp vs qbvh double -> opt-in,
// not bit-exact twin. Traversal order/narrowing mirrors qbvh.h spirit.
class qbvh8_node : public hittable {
public:
    qbvh8_node() {}
    qbvh8_node(std::vector<std::shared_ptr<hittable>> &objs, size_t start, size_t end,
              bool use_sah = true) {
        bvh_node tmp(objs, start, end, use_sah);
        box = tmp.node_box();
        collapse(tmp);
    }
    // Collapse existing binary tree (same DFS order).
    explicit qbvh8_node(const bvh_node &b) {
        box = b.node_box();
        collapse(b);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        count_box();
        count_box();
        count_box();
        count_box();
        count_box();
        count_box();
        count_box();
        count_box();
        unsigned mask = 0;
        double entry[8] = {t_max, t_max, t_max, t_max, t_max, t_max, t_max, t_max};
        mask = box_mask(r, t_min, t_max, entry);
        if (!mask)
            return false;
        hit_record tmp;
        bool any = false;
        double closest = t_max;
        int order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int norder = 0;
        order_by_entry(entry, mask, order, norder);
        for (int oi = 0; oi < norder; ++oi) {
            int s = order[oi];
            const child &c = kids[(size_t)s];
            if (c.leaf) {
                for (const auto &p : c.prims) {
                    ::count_prim();
                    tmp.hit_obj = nullptr; // stale medium tags (M48)
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
        ::count_box();
        ::count_box();
        ::count_box();
        ::count_box();
        unsigned mask = 0;
        double entry[8] = {t_max, t_max, t_max, t_max, t_max, t_max, t_max, t_max};
        mask = box_mask(r, t_min, t_max, entry);
        if (!mask)
            return false;
        int order[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int norder = 0;
        order_by_entry(entry, mask, order, norder);
        for (int oi = 0; oi < norder; ++oi) {
            int s = order[oi];
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

    // DFS prim expansion (same order as traversal).
    void collect_prims(std::vector<std::shared_ptr<hittable>> &out) const {
        for (int s = 0; s < nkids; ++s) {
            const child &c = kids[(size_t)s];
            if (c.leaf) {
                for (const auto &p : c.prims)
                    out.push_back(p);
            } else {
                c.node->collect_prims(out);
            }
        }
    }

private:
    struct child {
        aabb box;
        bool leaf = true;
        std::vector<std::shared_ptr<hittable>> prims; // leaf only
        std::unique_ptr<qbvh8_node> node; // inner only
    };

    aabb box;
    child kids[8];
    int nkids = 0;
    // Float SoA slabs. Empty slots inverted -> never hit.
    float mnx[8], mny[8], mnz[8], mxx[8], mxy[8], mxz[8];

    void add_leaf(const std::vector<std::shared_ptr<hittable>> &prims, const aabb &b) {
        if (nkids >= 8)
            return;
        child &c = kids[(size_t)nkids++];
        c.leaf = true;
        c.prims = prims;
        c.box = b;
        set_slab(nkids - 1, b);
    }

    void add_inner(const bvh_node &b) {
        if (nkids >= 8)
            return;
        child &c = kids[(size_t)nkids++];
        c.leaf = false;
        c.node = std::unique_ptr<qbvh8_node>(new qbvh8_node(b));
        c.box = b.node_box();
        set_slab(nkids - 1, c.box);
    }

    void set_slab(int s, const aabb &b) {
        mnx[s] = (float)b.minimum.x();
        mny[s] = (float)b.minimum.y();
        mnz[s] = (float)b.minimum.z();
        mxx[s] = (float)b.maximum.x();
        mxy[s] = (float)b.maximum.y();
        mxz[s] = (float)b.maximum.z();
    }

    static double box_area(const aabb &b) {
        double dx = b.maximum.x() - b.minimum.x();
        double dy = b.maximum.y() - b.minimum.y();
        double dz = b.maximum.z() - b.minimum.z();
        if (dx < 0 || dy < 0 || dz < 0)
            return 0.0;
        return 2.0 * (dx * dy + dy * dz + dz * dx);
    }

    // SAH-aware greedy 8-wide front: seed with the root's children, then
    // repeatedly expand the inner item with the largest area saving
    // SA(node) - SA(l) - SA(r) until 8 slots or no positive gain remains.
    // Beats purely structural collapse: big loose nodes split first, tight
    // small ones stay whole, so box tests cull more. DFS tie order keeps
    // traversal deterministic.
    void collapse(const bvh_node &b) {
        for (int i = 0; i < 8; ++i) {
            mnx[i] = 1e30f;
            mny[i] = 1e30f;
            mnz[i] = 1e30f;
            mxx[i] = -1e30f;
            mxy[i] = -1e30f;
            mxz[i] = -1e30f;
        }
        if (b.is_leaf()) {
            add_leaf(b.leaf_prims(), b.node_box());
            return;
        }
        struct Item {
            bool leaf = true;
            std::vector<std::shared_ptr<hittable>> prims;
            aabb box;
            const bvh_node *node = nullptr; // inner only
        };
        std::vector<Item> front;
        const bvh_node *lr[2] = {b.child(false).get(), b.child(true).get()};
        for (int k = 0; k < 2; ++k) {
            if (!lr[k])
                continue;
            Item it;
            it.box = lr[k]->node_box();
            if (lr[k]->is_leaf()) {
                it.leaf = true;
                it.prims = lr[k]->leaf_prims();
            } else {
                it.leaf = false;
                it.node = lr[k];
            }
            front.push_back(std::move(it));
        }
        for (;;) {
            if (front.size() >= 8)
                break;
            // Best inner expansion candidate (max area saving, DFS ties).
            // Fill to 8 regardless of gain sign: a full front halves depth
            // vs stopping early (overlapping children often score <= 0 but
            // expanding still culls better than an extra tree level).
            int best = -1;
            double best_gain = 0.0;
            for (size_t i = 0; i < front.size(); ++i) {
                if (front[i].leaf || !front[i].node)
                    continue;
                const bvh_node *l = front[i].node->child(false).get();
                const bvh_node *r = front[i].node->child(true).get();
                if (!l && !r)
                    continue;
                double gain = box_area(front[i].box);
                if (l)
                    gain -= box_area(l->node_box());
                if (r)
                    gain -= box_area(r->node_box());
                if (best < 0 || gain > best_gain) {
                    best = (int)i;
                    best_gain = gain;
                }
            }
            if (best < 0)
                break;
            const bvh_node *node = front[(size_t)best].node;
            const bvh_node *l = node->child(false).get();
            const bvh_node *r = node->child(true).get();
            front.erase(front.begin() + best);
            // Insert children at the same position (DFS order kept).
            int at = best;
            const bvh_node *kids[2] = {l, r};
            for (int k = 0; k < 2; ++k) {
                if (!kids[k])
                    continue;
                Item it;
                it.box = kids[k]->node_box();
                if (kids[k]->is_leaf()) {
                    it.leaf = true;
                    it.prims = kids[k]->leaf_prims();
                } else {
                    it.leaf = false;
                    it.node = kids[k];
                }
                front.insert(front.begin() + at++, std::move(it));
            }
        }
        for (auto &it : front) {
            if (nkids >= 8)
                break;
            if (it.leaf)
                add_leaf(it.prims, it.box);
            else if (it.node)
                add_inner(*it.node);
        }
    }

    // Slots visited near-first by slab entry. Stable: equal entries keep
    // DFS slot order.
    static void order_by_entry(const double entry[8], unsigned mask, int order[8],
                               int &count) {
        count = 0;
        for (int s = 0; s < 8; ++s)
            if (mask & (1u << (unsigned)s))
                order[count++] = s;
        for (int i = 1; i < count; ++i) {
            int key = order[i];
            int j = i - 1;
            while (j >= 0 && entry[order[j]] > entry[key]) {
                order[j + 1] = order[j];
                --j;
            }
            order[j + 1] = key;
        }
    }

    // 8-wide float slab test, single AVX pass: per-axis scalar invD sign
    // swap, compare+blend narrowing (NaN keeps old bound, like scalar),
    // strict miss. Scalar loop fallback when no AVX.
    // Entry distances are narrowed t_min values, for near-first sort.
    unsigned box_mask(const ray &r, double t_min, double t_max, double entry[8]) const {
#if defined(QBVH8_AVX)
        const vec3 &o = r.origin();
        const vec3 &inv = r.inv_direction();
        const __m256 o4[3] = {_mm256_set1_ps((float)o.x()), _mm256_set1_ps((float)o.y()),
                              _mm256_set1_ps((float)o.z())};
        const __m256 iv[3] = {_mm256_set1_ps((float)inv.x()), _mm256_set1_ps((float)inv.y()),
                              _mm256_set1_ps((float)inv.z())};
        const bool inv_neg[3] = {inv.x() < 0.0, inv.y() < 0.0, inv.z() < 0.0};
        const float *mns[3] = {mnx, mny, mnz};
        const float *mxs[3] = {mxx, mxy, mxz};
        __m256 tmn = _mm256_set1_ps((float)t_min);
        __m256 tmx = _mm256_set1_ps((float)t_max);
        for (int a = 0; a < 3; ++a) {
            __m256 lo = _mm256_loadu_ps(mns[a]);
            __m256 hi = _mm256_loadu_ps(mxs[a]);
            if (inv_neg[a]) {
                __m256 t = lo;
                lo = hi;
                hi = t;
            }
            __m256 t0 = _mm256_mul_ps(_mm256_sub_ps(lo, o4[a]), iv[a]);
            __m256 t1 = _mm256_mul_ps(_mm256_sub_ps(hi, o4[a]), iv[a]);
            __m256 gt0 = _mm256_cmp_ps(t0, tmn, _CMP_GT_OQ);
            tmn = _mm256_blendv_ps(tmn, t0, gt0);
            __m256 lt1 = _mm256_cmp_ps(t1, tmx, _CMP_LT_OQ);
            tmx = _mm256_blendv_ps(tmx, t1, lt1);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, tmn);
        for (int s = 0; s < 8; ++s)
            entry[s] = (double)tmp[s];
        return (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(tmx, tmn, _CMP_GT_OQ)) & 0xFFu;
#else
        unsigned mask = 0;
        for (int s = 0; s < nkids; ++s) {
            aabb b(vec3(mnx[s], mny[s], mnz[s]), vec3(mxx[s], mxy[s], mxz[s]));
            if (b.hit_entry(r, t_min, t_max, entry[s]))
                mask |= 1u << (unsigned)s;
        }
        return mask;
#endif
    }
};
