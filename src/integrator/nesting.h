#pragma once
// Nested-dielectric medium stack (M57): priority + prim-identity rules.
// Pure logic, no scene knowledge; mirrored in GLSL (path.comp Li locals).
// Depth cap 8; bottom frame is air (ior 1, pri -1, null id).
// M58: frames carry entry point + absorption; EXIT reports the chord so
// callers apply Beer's law for the traversed medium.
#include "../core/vec3.h"

#include <cmath>
#include <cstddef>

struct medium_frame {
    double ior = 1.0;
    int pri = -1;
    const void *id = nullptr; // owning prim (host: hittable*; device: id pair)
    vec3 entry{0, 0, 0};      // ENTER hit point (chord start)
    vec3 absorb{0, 0, 0};     // absorption sigma of the entered medium
};

class medium_stack {
public:
    static constexpr int CAP = 8;
    medium_stack() : n(1) { f[0] = medium_frame{}; }

    int depth() const { return n; }
    const medium_frame &top() const { return f[n - 1]; }

    enum event { ENTER, EXIT, PASS };
    // Interface event for a dielectric hit. Updates the stack, reports eta
    // (current_ior / next_ior). On EXIT also reports chord (|exit-entry|)
    // and the exited medium's absorption. PASS = legacy front_face
    // handling, stack untouched (lower-pri surface, same-pri overlap, or
    // overflow past CAP).
    event resolve(double surf_ior, int surf_pri, const void *surf_id,
                  const vec3 &hit_point, const vec3 &surf_absorb, double &eta,
                  double &chord, vec3 &exit_absorb) {
        const medium_frame &t = f[n - 1];
        if (surf_id != nullptr && surf_id == t.id && n > 1) {
            const medium_frame &ex = f[n - 1];
            vec3 dv = vec3(hit_point.x() - ex.entry.x(), hit_point.y() - ex.entry.y(),
                           hit_point.z() - ex.entry.z());
            eta = ex.ior / f[n - 2].ior;
            chord = dv.length();
            exit_absorb = ex.absorb;
            --n;
            return EXIT;
        }
        if (surf_pri > t.pri) {
            eta = t.ior / surf_ior;
            if (n < CAP) {
                medium_frame fr;
                fr.ior = surf_ior;
                fr.pri = surf_pri;
                fr.id = surf_id;
                fr.entry = hit_point;
                fr.absorb = surf_absorb;
                f[n] = fr;
                ++n;
                return ENTER;
            }
            return PASS; // overflow: legacy shade, stack untouched
        }
        return PASS;
    }

private:
    int n = 1;
    medium_frame f[CAP];
};

// Beer's law transmittance for one chord (per-channel).
inline vec3 beer_transmittance(const vec3 &sigma, double d) {
    return vec3(std::exp(-sigma.x() * d), std::exp(-sigma.y() * d),
                std::exp(-sigma.z() * d));
}
