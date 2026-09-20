// Shared scene structs + intersection. Included by normal/path kernels.
// Brute-force list (6 prims); GPU BVH flagged later like CPU M2->M5.
struct GPUSphere {
    vec4 c_r;
    vec4 c1;
    vec4 tm; // (t0, t1, *, *) motion range; c0==c1 when static
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, rough, ir, motionflag (fog: type 6, density)
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
};
struct GPUNode {
    vec4 bmin;
    vec4 bmax;
    ivec4 lrsc; // left, right, ref-start, ref-count (-1 = leaf)
};
struct GPURef {
    ivec2 ti; // (type, index): 0 sphere, 1 quad, 2 tri
};
struct GPUCam {
    vec4 origin;
    vec4 lower_left;
    vec4 horiz;
    vec4 vert;
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

bool hit_tri(vec3 o, vec3 d, float tmin, float tmax, GPUTri t_,
             out float t, out vec3 n, out vec2 uv) {
    const float eps = 1e-8;
    vec3 e1 = t_.b.xyz - t_.a.xyz;
    vec3 e2 = t_.c.xyz - t_.a.xyz;
    vec3 pvec = cross(d, e2);
    float det = dot(e1, pvec);
    if (abs(det) < eps)
        return false;
    float inv = 1.0 / det;
    vec3 tvec = o - t_.a.xyz;
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
    GPUNode nodes[];
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

// Iterative BVH walk, explicit 32-stack, left-first. Same closest-hit
// contract as brute force (narrowing tmax); fog slots pass through
// here (volume events come from fog_event, mirroring the CPU where
// the boundary never shades).
//
// Nearest fog event (type-6 slots) along the ray: boundary chord vs
// exponential sample with caller RNG. Mirrors CPU constant_medium:
// entry clamps to tmin (rays born inside still scatter), then exit.
bool fog_event(vec3 o, vec3 d, float rtime, int ns, float u01, float tmax,
               out float tevent, out vec3 talb) {
    tevent = 1e30;
    bool any = false;
    for (int i = 0; i < ns; ++i) {
        if (spheres[i].params.x != 6.0)
            continue;
        float te, tx;
        vec3 dn;
        vec2 duv;
        if (!hit_sphere(o, d, rtime, -1e30, tmax, spheres[i], te, dn, duv))
            continue;
        te = max(te, 0.001);
        if (!hit_sphere(o, d, rtime, te + 0.01, tmax, spheres[i], tx, dn, duv))
            continue;
        float s = -log(max(u01, 1e-7)) / max(spheres[i].params.y, 1e-7);
        if (s < tx - te && te + s < tevent) {
            tevent = te + s;
            talb = spheres[i].alb.xyz;
            any = true;
        }
    }
    return any;
}
// contract as the old brute loops (narrowing tmax), so kernels just swap.
void traverse(vec3 o, vec3 d, float rtime, float tmax, out float t, out vec3 n,
              out vec4 alb, out vec4 alb2, out vec4 emit, out vec4 params,
              out int light_idx, out vec2 huv, out bool any) {
    int stack[32];
    int sp = 0;
    stack[sp++] = 0;
    t = tmax;
    any = false;
    light_idx = -1;
    huv = vec2(0.0);
    while (sp > 0) {
        GPUNode nd = nodes[stack[--sp]];
        if (!hit_box(o, d, 0.001, t, nd.bmin.xyz, nd.bmax.xyz))
            continue;
        if (nd.lrsc.x < 0) {
            for (int k = 0; k < nd.lrsc.w; ++k) {
                GPURef ref = refs[nd.lrsc.z + k];
                float tt;
                vec3 nn;
                vec2 uv;
                if (ref.ti.x == 0) {
                    if (!hit_sphere(o, d, rtime, 0.001, t, spheres[ref.ti.y], tt, nn, uv))
                        continue;
                    GPUSphere s = spheres[ref.ti.y];
                    t = tt;
                    n = nn;
                    alb = s.alb;
                    alb2 = s.alb2;
                    emit = s.emit;
                    params = s.params;
                    huv = uv;
                    any = true;
                } else if (ref.ti.x == 1) {
                    if (!hit_quad(o, d, 0.001, t, quads[ref.ti.y], tt, nn, uv))
                        continue;
                    GPUQuad q = quads[ref.ti.y];
                    t = tt;
                    n = nn;
                    alb = q.alb;
                    alb2 = q.alb2;
                    emit = q.emit;
                    params = q.params;
                    huv = uv;
                    light_idx = (emit.x + emit.y + emit.z > 0.0) ? ref.ti.y : -1;
                    any = true;
                } else {
                    if (!hit_tri(o, d, 0.001, t, tris[ref.ti.y], tt, nn, uv))
                        continue;
                    GPUTri tr = tris[ref.ti.y];
                    t = tt;
                    n = nn;
                    alb = tr.alb;
                    alb2 = tr.alb2;
                    emit = tr.emit;
                    params = tr.params;
                    // Corner-UV blend when present, else barycentric fallback.
                    huv = (tr.params.w > 0.5)
                              ? tr.tuvA.xy * (1.0 - uv.x - uv.y) + tr.tuvA.zw * uv.x +
                                    tr.tuvB.xy * uv.y
                              : uv;
                    any = true;
                }
            }
        } else if (sp < 30) {
            stack[sp++] = nd.lrsc.y; // right
            stack[sp++] = nd.lrsc.x; // left pops first
        }
    }
}

// Solid-surface trace: fog slots pass through (up to 4 boundaries).
// Mirrors CPU where the medium boundary never shades. Returned t is
// absolute from o (travelled distance accumulated across passes).
bool trace_solid(vec3 o, vec3 d, float rtime, float tmax, out float t, out vec3 n,
                 out vec4 alb, out vec4 alb2, out vec4 emit, out vec4 params,
                 out int light_idx, out vec2 huv) {
    vec3 oo = o;
    float trav = 0.0;
    for (int k = 0; k < 4; ++k) {
        bool any;
        traverse(oo, d, rtime, tmax - trav, t, n, alb, alb2, emit, params, light_idx,
                 huv, any);
        if (!any)
            return false;
        if (params.x != 6.0) {
            t = trav + t;
            return true;
        }
        trav += t + 0.01;
        oo = o + d * trav;
    }
    return false;
}

// Shadow probe mirrors CPU: blocked by fog events OR solid surfaces
// before the light (medium competes by t on the CPU too).
bool shadow_occluded(vec3 o, vec3 wi, float rtime, int ns, float u01, float dist) {
    float fev;
    vec3 fa;
    if (fog_event(o, wi, rtime, ns, u01, dist - 0.001, fev, fa))
        return true;
    float t;
    vec3 n;
    vec4 alb, alb2, emit, params;
    int light_idx;
    vec2 huv;
    return trace_solid(o, wi, rtime, dist - 0.001, t, n, alb, alb2, emit, params,
                       light_idx, huv);
}
