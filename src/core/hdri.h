#pragma once
// HDRI equirectangular environment map with separable 2D importance sampling.
// Loads via stbi_loadf (HDR/EXR/float images); sRGB sources linearised by
// stb_loader. Builds a marginal CDF over rows and a conditional CDF per row,
// both weighted by sin(theta) so the solid-angle PDF is proportional to
// luminance regardless of the pole-compression artefact.
//
// GPU upload layout (cdf_upload()): flat float array stored as:
//   [0]           = float(W)
//   [1]           = float(H)
//   [2 .. H+2]    = marginal CDF (H+1 values; [2]=0, [H+2]=1)
//   [H+3 .. end]  = conditional CDFs, row r at offset H+3 + r*(W+1), W+1 vals
//
// Texels (texel_upload()): W*H vec4 (rgba32f), row 0 = top of image (v=0).
//
// UV ↔ direction convention matches the sphere UV used throughout the codebase:
//   u = (atan2(-z, x) + pi) / (2*pi)   [azimuth]
//   v = acos(y) / pi                    [polar, 0=top]
// Inverse (u,v) → direction:
//   theta = pi*v,  phi = 2*pi*u
//   dir = (-sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi))

#include "vec3.h"
// stbi_loadf is provided by the stb_image TU (src/io/stb_image.cpp).
// Include only the header here; the definition lives elsewhere.
#ifndef STBI_INCLUDE_STB_IMAGE_H
#include "../../external/stb/stb_image.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

class hdri_env {
public:
    int W = 0, H = 0;

    // Load from a Radiance RGBE .hdr file (or any stbi_loadf-supported format).
    // Returns false and leaves the object empty on failure.
    bool load(const std::string &path) {
        int ch = 0;
        float *raw = stbi_loadf(path.c_str(), &W, &H, &ch, 3);
        if (!raw || W <= 0 || H <= 0) {
            W = H = 0;
            return false;
        }
        pixels_.resize((size_t)W * H);
        for (size_t i = 0; i < (size_t)W * H; ++i)
            pixels_[i] = vec3(raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]);
        stbi_image_free(raw);
        build_cdf();
        return true;
    }

    bool empty() const { return pixels_.empty(); }

    // Bilinear sample of the HDR image for a given direction.
    vec3 radiance(const vec3 &d) const {
        auto [u, v] = dir_to_uv(d);
        return bilinear(u, v);
    }

    // Sample a direction importance-proportional-to-luminance*sinθ.
    // Returns direction (unit) and its solid-angle PDF.
    std::pair<vec3, double> sample(double u1, double u2) const {
        // Invert marginal CDF for row index.
        int vi = lower_bound_cdf(marginal_cdf_.data(), H, u1);
        double v0 = marginal_cdf_[vi], v1 = marginal_cdf_[vi + 1];
        double fv = (v1 > v0) ? (u1 - v0) / (v1 - v0) : 0.5;
        double vf = (vi + fv) / H;  // continuous [0,1]

        // Invert conditional CDF for column index given row.
        const float *cond = cond_cdf_.data() + (size_t)vi * (W + 1);
        int ui = lower_bound_cdf(cond, W, u2);
        double c0 = cond[ui], c1 = cond[ui + 1];
        double fu = (c1 > c0) ? (u2 - c0) / (c1 - c0) : 0.5;
        double uf = (ui + fu) / W;  // continuous [0,1]

        vec3 dir = uv_to_dir(uf, vf);

        double sin_theta = std::sqrt(std::max(1.0 - dir.y() * dir.y(), 0.0));
        // p(u,v) = (v1-v0)*H * (c1-c0)*W  (bin-normalised densities)
        // p(ω)   = p(u,v) / (2π² · sinθ)
        double pdf_uv = (v1 - v0) * H * (c1 - c0) * W;
        double pdf_w  = (sin_theta > 1e-5) ? pdf_uv / (2.0 * pi_ * pi_ * sin_theta) : 0.0;
        return {dir, pdf_w};
    }

    // Evaluate solid-angle PDF for an arbitrary direction (for MIS miss weight).
    double pdf(const vec3 &d) const {
        auto [u, v] = dir_to_uv(d);
        double sin_theta = std::sin(pi_ * v);
        if (sin_theta < 1e-5)
            return 0.0;

        int vi = std::min((int)(v * H), H - 1);
        double marg_diff = marginal_cdf_[vi + 1] - marginal_cdf_[vi];
        const float *cond = cond_cdf_.data() + (size_t)vi * (W + 1);
        int ui = std::min((int)(u * W), W - 1);
        double cond_diff = cond[ui + 1] - cond[ui];

        return marg_diff * cond_diff * W * H / (2.0 * pi_ * pi_ * sin_theta);
    }

    // --- GPU upload serialisers ---

    // Flat float CDF array: [W, H, marginal(H+1), conditionals(H*(W+1))].
    std::vector<float> cdf_upload() const {
        size_t n = 2 + (H + 1) + (size_t)H * (W + 1);
        std::vector<float> out;
        out.reserve(n);
        out.push_back((float)W);
        out.push_back((float)H);
        for (float v : marginal_cdf_)
            out.push_back(v);
        for (float v : cond_cdf_)
            out.push_back(v);
        return out;
    }

    // RGBA32f texel array, row 0 = top (v=0), as uploaded to the GPU.
    std::vector<float> texel_upload() const {
        std::vector<float> out;
        out.reserve((size_t)W * H * 4);
        for (const vec3 &p : pixels_) {
            out.push_back((float)p.x());
            out.push_back((float)p.y());
            out.push_back((float)p.z());
            out.push_back(1.0f);
        }
        return out;
    }

    // CDF accessors for unit tests.
    const std::vector<float> &marginal_cdf() const { return marginal_cdf_; }
    const std::vector<float> &cond_cdf()     const { return cond_cdf_; }

private:
    static constexpr double pi_ = 3.1415926535897932385;

    std::vector<vec3>  pixels_;       // W*H, row 0 = top
    std::vector<float> marginal_cdf_; // H+1 values
    std::vector<float> cond_cdf_;     // H*(W+1) values

    // Relative luminance (BT.709).
    static double lum(const vec3 &c) {
        return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
    }

    // Build separable CDF weighted by lum(u,v)*sin(theta_v).
    void build_cdf() {
        marginal_cdf_.resize((size_t)H + 1, 0.0f);
        cond_cdf_.resize((size_t)H * (W + 1), 0.0f);

        // Per-row sums (unnormalised).
        std::vector<double> row_sum(H, 0.0);
        for (int r = 0; r < H; ++r) {
            double theta = pi_ * (r + 0.5) / H;   // centre of row
            double sin_t = std::sin(theta);
            float *cond = cond_cdf_.data() + (size_t)r * (W + 1);
            double s = 0.0;
            cond[0] = 0.0f;
            for (int c = 0; c < W; ++c) {
                double w = lum(pixels_[(size_t)r * W + c]) * sin_t;
                if (w < 0.0) w = 0.0;
                s += w;
                cond[c + 1] = (float)s;
            }
            row_sum[r] = s;
        }

        // Normalise conditional CDFs.
        for (int r = 0; r < H; ++r) {
            float *cond = cond_cdf_.data() + (size_t)r * (W + 1);
            double rs = row_sum[r];
            if (rs > 0.0)
                for (int c = 0; c <= W; ++c)
                    cond[c] = (float)(cond[c] / rs);
            else
                for (int c = 0; c <= W; ++c)
                    cond[c] = (float)c / W;  // uniform fallback
        }

        // Build and normalise marginal CDF.
        double total = 0.0;
        marginal_cdf_[0] = 0.0f;
        for (int r = 0; r < H; ++r) {
            total += row_sum[r];
            marginal_cdf_[r + 1] = (float)total;
        }
        if (total > 0.0)
            for (int r = 0; r <= H; ++r)
                marginal_cdf_[r] = (float)(marginal_cdf_[r] / total);
        else
            for (int r = 0; r <= H; ++r)
                marginal_cdf_[r] = (float)r / H;  // uniform fallback
    }

    // Binary search in CDF[0..n] for the largest i where CDF[i] <= key.
    // Returns i in [0, n-1].
    static int lower_bound_cdf(const float *cdf, int n, double key) {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if ((double)cdf[mid + 1] <= key)
                lo = mid + 1;
            else
                hi = mid;
        }
        return std::max(0, std::min(lo, n - 1));
    }

    // Bilinear lookup on the HDR image.
    vec3 bilinear(double u, double v) const {
        if (pixels_.empty()) return vec3(0, 0, 0);
        double fx = u * (W - 1);
        double fy = (1.0 - v) * (H - 1);  // v=0 → top row = y=0
        int x0 = (int)fx, y0 = (int)fy;
        int x1 = std::min(x0 + 1, W - 1);
        int y1 = std::min(y0 + 1, H - 1);
        double tx = fx - x0, ty = fy - y0;
        const vec3 &c00 = pixels_[(size_t)y0 * W + x0];
        const vec3 &c10 = pixels_[(size_t)y0 * W + x1];
        const vec3 &c01 = pixels_[(size_t)y1 * W + x0];
        const vec3 &c11 = pixels_[(size_t)y1 * W + x1];
        return c00 * ((1 - tx) * (1 - ty)) + c10 * (tx * (1 - ty)) +
               c01 * ((1 - tx) * ty)       + c11 * (tx * ty);
    }

    // Direction → (u, v) in [0,1]² using the codebase sphere UV convention.
    static std::pair<double, double> dir_to_uv(const vec3 &d) {
        vec3 ud = unit_vector(d);
        // phi = atan2(-z, x) ∈ [-pi, pi]; u = (phi+pi) / (2pi)
        double phi = std::atan2(-ud.z(), ud.x());
        double u = (phi + pi_) / (2.0 * pi_);
        // v = acos(y) / pi; y clamped to [-1,1]
        double v = std::acos(std::max(-1.0, std::min(1.0, (double)ud.y()))) / pi_;
        return {u, v};
    }

    // (u, v) → unit direction (inverse of dir_to_uv).
    static vec3 uv_to_dir(double u, double v) {
        double theta = pi_ * v;
        double phi   = 2.0 * pi_ * u;
        double sin_t = std::sin(theta);
        // From dir_to_uv: phi_raw = atan2(-z, x) → x = sin_t*cos(phi-pi), z = -sin_t*sin(phi-pi)
        // cos(phi-pi) = -cos(phi), sin(phi-pi) = -sin(phi)
        // → x = -sin_t*cos(phi), z = sin_t*sin(phi), y = cos(theta)
        double x = -sin_t * std::cos(phi);
        double y =  std::cos(theta);
        double z =  sin_t * std::sin(phi);
        return unit_vector(vec3(x, y, z));
    }
};
