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
              out float t, out vec3 n) {
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
    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
        return false;
    t = tt;
    n = (dot(d, nrm) > 0.0) ? -nrm : nrm;
    return true;
}
