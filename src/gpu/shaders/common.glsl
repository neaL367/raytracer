// Shared scene structs + intersection. Included by normal/path kernels.
// 4-wide QBVH traversal over the same SAH collapse the CPU walks.

// Material types matching C++ MatType enum class
const int MAT_SOLID     = 0;
const int MAT_RESERVED  = 1;
const int MAT_GLASS     = 2;
const int MAT_EMIT      = 3;
const int MAT_CHECKER   = 4;
const int MAT_IMAGE     = 5;
const int MAT_FOG       = 6;
const int MAT_CONDUCTOR = 7;
const int MAT_HET       = 8;
const int MAT_NOISE     = 9;
const int MAT_ANISO     = 10;
const int MAT_DISNEY    = 11;
struct GPUSphere {
    vec4 c_r;
    vec4 c1;
    vec4 tm; // (t0, t1, *, *) motion range; c0==c1 when static
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, rough, ir, motionflag (fog: type 6/8, density)
};
struct GPUQuad {
    vec4 Q;
    vec4 u;
    vec4 v;
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, rough, ir, 0
};
struct GPUTri {
    vec4 a;
    vec4 b;
    vec4 c;
    vec4 n0;
    vec4 n1;
    vec4 n2;
    vec4 tuvA; // (u0,v0,u1,v1) corner UVs
    vec4 tuvB; // (u2,v2,*,*) corner UVs
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, rough/ir/scale, ir, has_uv
    vec4 a1;
    vec4 b1;
    vec4 c1;
    vec4 tm; // motion range (moving when tm.y > tm.x)
};
struct GPUQNode {
    vec4 qbmin[4];
    vec4 qbmax[4];
    ivec4 qchild; // node index, or -1 leaf
    ivec4 qstart; // ref-run start per slot (leaf)
    ivec4 qcount; // ref-run count per slot (leaf)
};
struct GPURef {
    ivec2 ti; // (type, index): 0 sphere, 1 quad, 2 tri
};
struct GPUCam {
    vec4 origin;
    vec4 lower_left;
    vec4 horiz;
    vec4 vert;
    vec4 lens; // x = thin-lens radius (0 = pinhole)
};

bool hit_sphere(vec3 o, vec3 d, float rtime, float tmin, float tmax, GPUSphere s,
                out float t, out vec3 n, out vec2 uv) {
    // Motion lerp mirrors CPU (clamped); static spheres no-op (c0==c1).
    float f = (s.tm.y > s.tm.x) ? clamp((rtime - s.tm.x) / (s.tm.y - s.tm.x), 0.0, 1.0)
                                : 0.0;
    vec3 cen = mix(s.c_r.xyz, s.c1.xyz, f);
    vec3 oc = o - cen;
    float a = dot(d, d);
    float hb = dot(oc, d);
    float c = dot(oc, oc) - s.c_r.w * s.c_r.w;
    float disc = hb * hb - a * c;
    if (disc < 0.0)
        return false;
    float sq = sqrt(disc);
    float root = (-hb - sq) / a;
    if (root < tmin || root > tmax) {
        root = (-hb + sq) / a;
        if (root < tmin || root > tmax)
            return false;
    }
    t = root;
    n = (o + d * root - cen) / s.c_r.w;
    // Spherical UVs mirror the CPU (azimuth u, polar v).
    vec3 op = (o + d * root - cen) / s.c_r.w;
    uv = vec2((atan(-op.z, op.x) + 3.14159265) / 6.2831853,
              acos(clamp(op.y, -1.0, 1.0)) / 3.14159265);
    return true;
}

bool hit_tri(vec3 o, vec3 d, float rtime, float tmin, float tmax, GPUTri t_,
             out float t, out vec3 n, out vec2 uv) {
    const float eps = 1e-8;
    // Motion lerp mirrors CPU vert_at (clamped); static tris no-op.
    float f = (t_.tm.y > t_.tm.x)
                  ? clamp((rtime - t_.tm.x) / (t_.tm.y - t_.tm.x), 0.0, 1.0)
                  : 0.0;
    vec3 va = mix(t_.a.xyz, t_.a1.xyz, f);
    vec3 vb = mix(t_.b.xyz, t_.b1.xyz, f);
    vec3 vc = mix(t_.c.xyz, t_.c1.xyz, f);
    vec3 e1 = vb - va, e2 = vc - va;
    vec3 pvec = cross(d, e2);
    float det = dot(e1, pvec);
    if (abs(det) < eps)
        return false;
    float inv = 1.0 / det;
    vec3 tvec = o - va.xyz;
    float u = dot(tvec, pvec) * inv;
    if (u < 0.0 || u > 1.0)
        return false;
    vec3 qvec = cross(tvec, e1);
    float v = dot(d, qvec) * inv;
    if (v < 0.0 || u + v > 1.0)
        return false;
    float tt = dot(e2, qvec) * inv;
    if (tt < tmin || tt > tmax)
        return false;
    t = tt;
    // Smooth normals when present, else face normal: barycentric blend
    // mirrors the CPU (flat tris upload face normal x3, same result).
    vec3 face_n = normalize(cross(e1, e2));
    vec3 blend = normalize(t_.n0.xyz * (1.0 - u - v) + t_.n1.xyz * u + t_.n2.xyz * v);
    vec3 outward = (dot(blend, face_n) < 0.0) ? -blend : blend;
    n = (dot(d, outward) > 0.0) ? -outward : outward;
    uv = vec2(u, v); // barycentric, mirrors CPU
    return true;
}

bool hit_quad(vec3 o, vec3 d, float tmin, float tmax, GPUQuad q,
              out float t, out vec3 n, out vec2 uv) {    vec3 nrm = normalize(cross(q.u.xyz, q.v.xyz));
    float denom = dot(nrm, d);
    if (abs(denom) < 1e-8)
        return false;
    float D = dot(nrm, q.Q.xyz);
    float tt = (D - dot(nrm, o)) / denom;
    if (tt < tmin || tt > tmax)
        return false;
    vec3 p = o + d * tt;
    vec3 w = cross(q.u.xyz, q.v.xyz);
    w = w / dot(w, w);
    vec3 pq = p - q.Q.xyz;
    float alpha = dot(w, cross(pq, q.v.xyz));
    float beta = dot(w, cross(q.u.xyz, pq));
    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
        return false;
    t = tt;
    n = (dot(d, nrm) > 0.0) ? -nrm : nrm;
    uv = vec2(alpha, beta); // parametric, mirrors CPU
    return true;
}

// Shared scene buffers (both kernels declare image/UBO/push themselves).
layout(binding = 2) readonly buffer Spheres {
    GPUSphere spheres[];
};
layout(binding = 3) readonly buffer Quads {
    GPUQuad quads[];
};
layout(binding = 4) readonly buffer Tris {
    GPUTri tris[];
};
layout(binding = 5) readonly buffer Nodes {
    GPUQNode nodes[];
};
layout(binding = 6) readonly buffer Refs {
    GPURef refs[];
};

bool hit_box(vec3 o, vec3 d, float tmin, float tmax, vec3 bmin, vec3 bmax) {
    vec3 inv = 1.0 / d;
    vec3 t0 = (bmin - o) * inv;
    vec3 t1 = (bmax - o) * inv;
    vec3 tsm = min(t0, t1);
    vec3 tbg = max(t0, t1);
    float mn = max(max(tsm.x, tsm.y), max(tsm.z, tmin));
    float mx = min(min(tbg.x, tbg.y), min(tbg.z, tmax));
    return mx > mn;
}

// Iterative 4-wide QBVH walk, explicit 32-stack, slot 0 pops first (DFS,
// mirroring the CPU flat mirror). Same closest-hit contract as before
// (narrowing tmax); fog slots pass through here (volume events come from
// fog_event, mirroring the CPU where the boundary never shades).
// skip_ty/skip_idx/skip_t exclude the ray origin prim nearer than skip_t
// (fp32 self-skims; M54): single-pass, no re-walk. (-1,-1) = none.
//
// Counter-based hash: independent uniforms per (base, bounce, step, tag).
// Delta tracking must not chain one xorshift stream (consecutive-pair
// lattice biases acceptance); each tentative draws fresh hashes.
uint h32(uint x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16u)) * 0x21f0aaadu;
    x = (x ^ (x >> 15u)) * 0x735a2d97u;
    return x ^ (x >> 15u);
}
float h32f(uint base, int bounce, int site, int k, int tag) {
    uint h = base ^ (uint(bounce) * 0x85EBCA6Bu) ^ (uint(site) * 0xC2B2AE35u) ^
             (uint(k) * 2u + uint(tag));
    return float(h32(h)) / 4294967296.0;
}
bool fog_event(vec3 o, vec3 d, float rtime, int ns, float u01, float tmax,
                out float tevent, out vec3 talb, uint hbase, int bounce, int site,
                out int slot) {
    tevent = 1e30;
    slot = -1;
    bool any = false;
    for (int i = 0; i < ns; ++i) {
        float mtype = spheres[i].params.x;
        if (mtype != float(MAT_FOG) && mtype != float(MAT_HET))
            continue;
        float te, tx;
        vec3 dn;
        vec2 duv;
        if (!hit_sphere(o, d, rtime, -1e30, tmax, spheres[i], te, dn, duv))
            continue;
        te = max(te, 0.001);
        // Exit search is unbounded, then clamped (mirrors CPU): a border
        // past the light still fills the ray up to the light. Bounding the
        // search by tmax instead drops those events and leaks lights.
        // Offset mirrors CPU (1e-4): coarser steps skip close exits.
        if (!hit_sphere(o, d, rtime, te + 1e-4, 1e30, spheres[i], tx, dn, duv))
            continue;
        tx = min(tx, tmax);
        if (mtype == float(MAT_FOG)) {
            float s = -log(max(u01, 1e-7)) / max(spheres[i].params.y, 1e-7);
            if (s < tx - te && te + s < tevent) {
                tevent = te + s;
                talb = spheres[i].alb.xyz;
                slot = i;
                any = true;
            }
        } else {
            // Delta tracking at the majorant; accept on modulation.
            // Per-slot hash base (multi-slot scenes stay decorrelated).
            float sig = max(spheres[i].params.y, 1e-7);
            vec3 fr = spheres[i].alb2.xyz;
            uint sbase = h32(hbase ^ (uint(i) * 0x9E3779B9u));
            float cursor = te;
            for (int k = 0; k < 1024; ++k) {
                float s = -log(max(h32f(sbase, bounce, site, k, 0), 1e-7)) / sig;
                float x = cursor + s;
                if (x > tx)
                    break;
                vec3 p = o + d * x;
                float m = 0.5 + 0.5 * sin(fr.x * p.x) * sin(fr.y * p.y) *
                                            sin(fr.z * p.z);
                if (m > h32f(sbase, bounce, site, k, 1)) {
                    if (x < tevent) {
                        tevent = x;
                        talb = spheres[i].alb.xyz;
                        slot = i;
                        any = true;
                    }
                    break;
                }
                cursor = x;
            }
        }
    }
    return any;
}

// Fast boolean-only intersection tests for shadow occlusion
bool hit_sphere_test(vec3 o, vec3 d, float rtime, float tmin, float tmax, GPUSphere s) {
    float f = (s.tm.y > s.tm.x) ? clamp((rtime - s.tm.x) / (s.tm.y - s.tm.x), 0.0, 1.0)
                                : 0.0;
    vec3 cen = mix(s.c_r.xyz, s.c1.xyz, f);
    vec3 oc = o - cen;
    float a = dot(d, d);
    float hb = dot(oc, d);
    float c = dot(oc, oc) - s.c_r.w * s.c_r.w;
    float disc = hb * hb - a * c;
    if (disc < 0.0)
        return false;
    float sq = sqrt(disc);
    float root = (-hb - sq) / a;
    if (root >= tmin && root <= tmax)
        return true;
    root = (-hb + sq) / a;
    return (root >= tmin && root <= tmax);
}

bool hit_quad_test(vec3 o, vec3 d, float tmin, float tmax, GPUQuad q) {
    vec3 nrm = normalize(cross(q.u.xyz, q.v.xyz));
    float denom = dot(nrm, d);
    if (abs(denom) < 1e-8)
        return false;
    float D = dot(nrm, q.Q.xyz);
    float tt = (D - dot(nrm, o)) / denom;
    if (tt < tmin || tt > tmax)
        return false;
    vec3 p = o + d * tt;
    vec3 w = cross(q.u.xyz, q.v.xyz);
    w = w / dot(w, w);
    vec3 pq = p - q.Q.xyz;
    float alpha = dot(w, cross(pq, q.v.xyz));
    float beta = dot(w, cross(q.u.xyz, pq));
    return (alpha >= 0.0 && alpha <= 1.0 && beta >= 0.0 && beta <= 1.0);
}

bool hit_tri_test(vec3 o, vec3 d, float rtime, float tmin, float tmax, GPUTri t_) {
    const float eps = 1e-8;
    float f = (t_.tm.y > t_.tm.x)
                  ? clamp((rtime - t_.tm.x) / (t_.tm.y - t_.tm.x), 0.0, 1.0)
                  : 0.0;
    vec3 va = mix(t_.a.xyz, t_.a1.xyz, f);
    vec3 vb = mix(t_.b.xyz, t_.b1.xyz, f);
    vec3 vc = mix(t_.c.xyz, t_.c1.xyz, f);
    vec3 e1 = vb - va, e2 = vc - va;
    vec3 pvec = cross(d, e2);
    float det = dot(e1, pvec);
    if (abs(det) < eps)
        return false;
    float inv = 1.0 / det;
    vec3 tvec = o - va.xyz;
    float u = dot(tvec, pvec) * inv;
    if (u < 0.0 || u > 1.0)
        return false;
    vec3 qvec = cross(tvec, e1);
    float v = dot(d, qvec) * inv;
    if (v < 0.0 || u + v > 1.0)
        return false;
    float tt = dot(e2, qvec) * inv;
    return (tt >= tmin && tt <= tmax);
}

// Fast any-hit occlusion test for shadow rays. Aborts on the first solid hit.
// Fog slots (params.x == 6.0 || params.x == 8.0) pass through because volume
// transmittance is accounted for separately in shadow_transmittance.
bool is_occluded(vec3 o, vec3 d, float rtime, float tmax, int skip_ty, int skip_idx) {
    int stack[16];
    int sp = 0;
    stack[sp++] = 0;
    vec3 inv = 1.0 / d;
    while (sp > 0) {
        GPUQNode nd = nodes[stack[--sp]];
        int mask = 0;
        for (int s = 0; s < 4; ++s) {
            vec3 t0 = (nd.qbmin[s].xyz - o) * inv;
            vec3 t1 = (nd.qbmax[s].xyz - o) * inv;
            vec3 tsm = min(t0, t1);
            vec3 tbg = max(t0, t1);
            float mn = max(max(tsm.x, tsm.y), max(tsm.z, 0.001));
            float mx = min(min(tbg.x, tbg.y), min(tbg.z, tmax));
            if (mx > mn)
                mask |= (1 << s);
        }
        for (int s = 3; s >= 0; --s) {
            if ((mask & (1 << s)) == 0)
                continue;
            if (nd.qchild[s] < 0) {
                int start = nd.qstart[s];
                int count = nd.qcount[s];
                for (int k = 0; k < count; ++k) {
                    GPURef ref = refs[start + k];
                    if (ref.ti.x == skip_ty && ref.ti.y == skip_idx)
                        continue;
                    if (ref.ti.x == 0) {
                        float mtype = spheres[ref.ti.y].params.x;
                        if (mtype == 6.0 || mtype == 8.0)
                            continue;
                        if (hit_sphere_test(o, d, rtime, 0.001, tmax, spheres[ref.ti.y]))
                            return true;
                    } else if (ref.ti.x == 1) {
                        if (hit_quad_test(o, d, 0.001, tmax, quads[ref.ti.y]))
                            return true;
                    } else {
                        if (hit_tri_test(o, d, rtime, 0.001, tmax, tris[ref.ti.y]))
                            return true;
                    }
                }
            } else if (sp < 15) {
                stack[sp++] = nd.qchild[s];
            }
        }
    }
    return false;
}

// 4-wide QBVH traversal: 4 child AABBs tested in parallel, DFS stack. Same
// contract as the old brute loops (narrowing tmax), so kernels just swap.
void traverse(vec3 o, vec3 d, float rtime, float tmax, out float t, out vec3 n,
              out vec4 alb, out vec4 alb2, out vec4 emit, out vec4 params,
               out int light_idx, out int light_ty, out vec2 huv,
               out vec3 tang, out bool has_tang, out vec3 geo_n, out bool any,
               out int solid_ty, out int solid_idx,
               int skip_ty, int skip_idx, float skip_t) {
    int stack[16];
    int sp = 0;
    stack[sp++] = 0;
    t = tmax;
    any = false;
    solid_ty = -1;
    solid_idx = -1;
    light_idx = -1;
    light_ty = -1;
    huv = vec2(0.0);
    has_tang = false;
    geo_n = vec3(0.0, 1.0, 0.0);
    vec3 inv = 1.0 / d;
    while (sp > 0) {
        GPUQNode nd = nodes[stack[--sp]];
        // 4-wide slab: one interval per slot, strict miss, empty slots
        // upload inverted (never hit).
        int mask = 0;
        for (int s = 0; s < 4; ++s) {
            vec3 t0 = (nd.qbmin[s].xyz - o) * inv;
            vec3 t1 = (nd.qbmax[s].xyz - o) * inv;
            vec3 tsm = min(t0, t1);
            vec3 tbg = max(t0, t1);
            float mn = max(max(tsm.x, tsm.y), max(tsm.z, 0.001));
            float mx = min(min(tbg.x, tbg.y), min(tbg.z, t));
            if (mx > mn)
                mask |= (1 << s);
        }
        // Push high-to-low so slot 0 pops first (DFS order).
        for (int s = 3; s >= 0; --s) {
            if ((mask & (1 << s)) == 0)
                continue;
            if (nd.qchild[s] < 0) {
                for (int k = 0; k < nd.qcount[s]; ++k) {
                    GPURef ref = refs[nd.qstart[s] + k];
                float tt;
                vec3 nn;
                vec2 uv;
                if (ref.ti.x == 0) {
                    // Fog slots never shade as surfaces: volume events come
                    // from fog_event, transmittance from the shadow march.
                    // (M51: a co-located fog slot processed after a solid in
                    // the same leaf run overwrote its params, making glass
                    // shells coincident with smoke invisible. Skip up front.)
                    if (spheres[ref.ti.y].params.x == 6.0 ||
                        spheres[ref.ti.y].params.x == 8.0)
                        continue;
                    if (!hit_sphere(o, d, rtime, 0.001, t, spheres[ref.ti.y], tt, nn, uv))
                        continue;
                    // Origin self-skim (M54): fp32 re-hit past tmin of the
                    // surface the ray leaves; skip inline (no re-walk).
                    if (ref.ti.x == skip_ty && ref.ti.y == skip_idx && tt < skip_t)
                        continue;
                    t = tt;
                    n = nn;
                    huv = uv;
                    solid_ty = 0;
                    solid_idx = ref.ti.y;
                    any = true;
                } else if (ref.ti.x == 1) {
                    if (!hit_quad(o, d, 0.001, t, quads[ref.ti.y], tt, nn, uv))
                        continue;
                    if (ref.ti.x == skip_ty && ref.ti.y == skip_idx && tt < skip_t)
                        continue;
                    t = tt;
                    n = nn;
                    huv = uv;
                    solid_ty = 1;
                    solid_idx = ref.ti.y;
                    any = true;
                } else {
                    if (!hit_tri(o, d, rtime, 0.001, t, tris[ref.ti.y], tt, nn, uv))
                        continue;
                    if (ref.ti.x == skip_ty && ref.ti.y == skip_idx && tt < skip_t)
                        continue;
                    t = tt;
                    n = nn;
                    huv = uv;
                    solid_ty = 2;
                    solid_idx = ref.ti.y;
                    any = true;
                }
            }
        } else if (sp < 15) {
            stack[sp++] = nd.qchild[s];
        }
    }
}

    if (any) {
        if (solid_ty == 0) {
            geo_n = n;
            GPUSphere s = spheres[solid_idx];
            alb = s.alb;
            alb2 = s.alb2;
            emit = s.emit;
            params = s.params;
            vec3 st = vec3(-n.z, 0.0, n.x);
            if (dot(st, st) <= 1e-12)
                st = vec3(1.0, 0.0, 0.0);
            else
                st = normalize(st);
            tang = st;
            has_tang = true;
            bool fogslot = params.x > 5.5 && params.x < 6.5;
            bool emit_image = params.x == float(MAT_EMIT) && params.y > 0.5;
            bool lit = !fogslot && (emit.x + emit.y + emit.z > 0.0 || emit_image);
            light_idx = lit ? solid_idx : -1;
            light_ty = lit ? 1 : -1;
        } else if (solid_ty == 1) {
            geo_n = n;
            GPUQuad q = quads[solid_idx];
            alb = q.alb;
            alb2 = q.alb2;
            emit = q.emit;
            params = q.params;
            vec3 qg = normalize(cross(q.u.xyz, q.v.xyz));
            tang = normalize(q.u.xyz - qg * dot(q.u.xyz, qg));
            has_tang = true;
            bool q_emit_image = params.x == float(MAT_EMIT) && params.y > 0.5;
            bool q_lit = emit.x + emit.y + emit.z > 0.0 || q_emit_image;
            light_idx = q_lit ? solid_idx : -1;
            light_ty = q_lit ? 0 : -1;
        } else {
            GPUTri tr = tris[solid_idx];
            alb = tr.alb;
            alb2 = tr.alb2;
            emit = tr.emit;
            params = tr.params;
            vec2 uv = huv;
            huv = (tr.params.w > 0.5)
                      ? tr.tuvA.xy * (1.0 - uv.x - uv.y) + tr.tuvA.zw * uv.x +
                            tr.tuvB.xy * uv.y
                      : uv;
            float tf =
                (tr.tm.y > tr.tm.x)
                    ? clamp((rtime - tr.tm.x) / (tr.tm.y - tr.tm.x), 0.0, 1.0)
                    : 0.0;
            vec3 va = mix(tr.a.xyz, tr.a1.xyz, tf);
            vec3 vb = mix(tr.b.xyz, tr.b1.xyz, tf);
            vec3 vc = mix(tr.c.xyz, tr.c1.xyz, tf);
            vec3 te1 = vb - va, te2 = vc - va;
            vec3 tface = normalize(cross(te1, te2));
            geo_n = (dot(d, tface) > 0.0) ? -tface : tface;
            has_tang = false;
            if (tr.params.w > 0.5) {
                vec2 td1 = vec2(tr.tuvA.z - tr.tuvA.x, tr.tuvA.w - tr.tuvA.y);
                vec2 td2 = vec2(tr.tuvB.x - tr.tuvA.x, tr.tuvB.y - tr.tuvA.y);
                float tdet = td1.x * td2.y - td2.x * td1.y;
                if (abs(tdet) > 1e-12) {
                    vec3 ttv = (te1 * td2.y - te2 * td1.y) / tdet;
                    ttv = ttv - tface * dot(ttv, tface);
                    if (dot(ttv, ttv) > 1e-12) {
                        tang = normalize(ttv);
                        has_tang = true;
                    }
                }
            } else {
                bool is_smooth = dot(tr.n0.xyz - tr.n1.xyz, tr.n0.xyz - tr.n1.xyz) > 1e-6 ||
                                 dot(tr.n1.xyz - tr.n2.xyz, tr.n1.xyz - tr.n2.xyz) > 1e-6;
                if (!is_smooth) {
                    vec3 ttv = te1 - tface * dot(te1, tface);
                    if (dot(ttv, ttv) > 1e-12) {
                        tang = normalize(ttv);
                        has_tang = true;
                    }
                }
            }
            bool tr_emit_image = params.x == float(MAT_EMIT) && params.y > 0.5;
            bool trlit = emit.x + emit.y + emit.z > 0.0 || tr_emit_image;
            light_idx = trlit ? solid_idx : -1;
            light_ty = trlit ? 2 : -1;
        }
    }
}

// Solid-surface trace: fog slots pass through (up to 4 boundaries).
// Mirrors CPU where the medium boundary never shades. Returned t is
// absolute from o (travelled distance accumulated across passes).
// skip_ty/skip_idx (-1,-1 = none) excludes the ray origin prim: fp32
// self re-hits (computed t past tmin, true t below it) would otherwise
// shadow/block. skip_t gates it: origin hits nearer than skip_t step
// past (self-skim band); farther ones accept (glass transmission exits,
// TIR far sides). Shadows pass 1e30 (unconditional: outward diffuse
// rays never legitimately re-hit); beauty bounces pass 0.5. The 0.5
// covers inside-exit chords from fp32-short hit points (2*sqrt(2*r*d)
// with r=70 glass, d~2e-4 t-error: ~0.34); real origin chords below it
// are measure-zero grazing transits.
bool trace_solid(vec3 o, vec3 d, float rtime, float tmax, out float t, out vec3 n,
                 out vec4 alb, out vec4 alb2, out vec4 emit, out vec4 params,
                 out int light_idx, out int light_ty, out vec2 huv, out vec3 tang,
                 out bool has_tang, out vec3 geo_n, int skip_ty, int skip_idx, float skip_t,
                 out int solid_ty, out int solid_idx) {
    bool any;
    traverse(o, d, rtime, tmax, t, n, alb, alb2, emit, params, light_idx,
             light_ty, huv, tang, has_tang, geo_n, any, solid_ty, solid_idx,
             skip_ty, skip_idx, skip_t);
    return any;
}

// NEE shadow transmittance: solid in range -> 0; else product of per-slot
// segments (type-6 analytic, type-8 ratio-tracked). Mirrors the CPU march.
// Draw-free (counter hashes / analytic): shadow calls consume no RNG.
float shadow_transmittance(vec3 o, vec3 wi, float rtime, int ns, float dist,
                           uint hbase, int bounce, int site,
                           int skip_ty, int skip_idx) {
    float tmax = dist - 0.001;
    if (is_occluded(o, wi, rtime, tmax, skip_ty, skip_idx))
        return 0.0;
    float Tr = 1.0;
    for (int i = 0; i < ns; ++i) {
        float mtype = spheres[i].params.x;
        if (mtype != float(MAT_FOG) && mtype != float(MAT_HET))
            continue;
        float te, tx;
        vec3 dn;
        vec2 duv;
        if (!hit_sphere(o, wi, rtime, -1e30, tmax, spheres[i], te, dn, duv))
            continue;
        te = max(te, 0.001);
        // Exit search offset mirrors CPU (1e-4): a coarser step would skip
        // real exits just ahead of rays born inside the boundary.
        if (!hit_sphere(o, wi, rtime, te + 1e-4, 1e30, spheres[i], tx, dn, duv))
            continue;
        tx = min(tx, tmax);
        if (tx <= te)
            continue;
        if (mtype == float(MAT_FOG)) {
            Tr *= exp(-max(spheres[i].params.y, 1e-7) * (tx - te));
        } else {
            float sig = max(spheres[i].params.y, 1e-7);
            vec3 fr = spheres[i].alb2.xyz;
            uint sbase = h32(hbase ^ (uint(i) * 0x9E3779B9u));
            float cursor = te;
            for (int k = 0; k < 1024; ++k) {
                float s = -log(max(h32f(sbase, bounce, site, k, 0), 1e-7)) / sig;
                float x = cursor + s;
                if (x > tx)
                    break;
                vec3 p = o + wi * x;
                float m = 0.5 + 0.5 * sin(fr.x * p.x) * sin(fr.y * p.y) *
                                                sin(fr.z * p.z);
                Tr *= 1.0 - m;
                if (Tr <= 0.0)
                    return 0.0;
                cursor = x;
            }
        }
        if (Tr <= 0.0)
            return 0.0;
    }
    return Tr;
}
