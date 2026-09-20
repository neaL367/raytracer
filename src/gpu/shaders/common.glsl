// Shared scene structs + intersection. Included by normal/path kernels.
// Brute-force list (6 prims); GPU BVH flagged later like CPU M2->M5.
struct GPUSphere {
    vec4 c_r;
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, fuzz, ir, 0
};
struct GPUQuad {
    vec4 Q;
    vec4 u;
    vec4 v;
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, fuzz, ir, 0
};
struct GPUTri {
    vec4 a;
    vec4 b;
    vec4 c;
    vec4 alb;
    vec4 alb2;
    vec4 emit;
    vec4 params; // mat_type, fuzz, ir, 0
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

bool hit_sphere(vec3 o, vec3 d, float tmin, float tmax, GPUSphere s,
                out float t, out vec3 n) {
    vec3 oc = o - s.c_r.xyz;
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
    n = (o + d * root - s.c_r.xyz) / s.c_r.w;
    return true;
}

bool hit_tri(vec3 o, vec3 d, float tmin, float tmax, GPUTri t_,
             out float t, out vec3 n) {
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
    n = normalize(cross(e1, e2));
    // Double-sided: flip toward ray like CPU set_face_normal.
    if (dot(d, n) > 0.0)
        n = -n;
    return true;
}

bool hit_quad(vec3 o, vec3 d, float tmin, float tmax, GPUQuad q,
              out float t, out vec3 n) {    vec3 nrm = normalize(cross(q.u.xyz, q.v.xyz));
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
// contract as the old brute loops (narrowing tmax), so kernels just swap.
void traverse(vec3 o, vec3 d, float tmax, out float t, out vec3 n, out vec4 alb,
              out vec4 alb2, out vec4 emit, out vec4 params, out int light_idx,
              out bool any) {
    int stack[32];
    int sp = 0;
    stack[sp++] = 0;
    t = tmax;
    any = false;
    light_idx = -1;
    while (sp > 0) {
        GPUNode nd = nodes[stack[--sp]];
        if (!hit_box(o, d, 0.001, t, nd.bmin.xyz, nd.bmax.xyz))
            continue;
        if (nd.lrsc.x < 0) {
            for (int k = 0; k < nd.lrsc.w; ++k) {
                GPURef ref = refs[nd.lrsc.z + k];
                float tt;
                vec3 nn;
                if (ref.ti.x == 0) {
                    if (!hit_sphere(o, d, 0.001, t, spheres[ref.ti.y], tt, nn))
                        continue;
                    GPUSphere s = spheres[ref.ti.y];
                    t = tt;
                    n = nn;
                    alb = s.alb;
                    alb2 = s.alb2;
                    emit = s.emit;
                    params = s.params;
                    any = true;
                } else if (ref.ti.x == 1) {
                    if (!hit_quad(o, d, 0.001, t, quads[ref.ti.y], tt, nn))
                        continue;
                    GPUQuad q = quads[ref.ti.y];
                    t = tt;
                    n = nn;
                    alb = q.alb;
                    alb2 = q.alb2;
                    emit = q.emit;
                    params = q.params;
                    light_idx = (emit.x + emit.y + emit.z > 0.0) ? ref.ti.y : -1;
                    any = true;
                } else {
                    if (!hit_tri(o, d, 0.001, t, tris[ref.ti.y], tt, nn))
                        continue;
                    GPUTri tr = tris[ref.ti.y];
                    t = tt;
                    n = nn;
                    alb = tr.alb;
                    alb2 = tr.alb2;
                    emit = tr.emit;
                    params = tr.params;
                    any = true;
                }
            }
        } else if (sp < 30) {
            stack[sp++] = nd.lrsc.y; // right
            stack[sp++] = nd.lrsc.x; // left pops first
        }
    }
}
