#pragma once
// Nested-dielectric medium stack (M57): priority + prim-identity rules.
// Pure logic, no scene knowledge; mirrored in GLSL (path.comp Li locals).
// Depth cap 8; bottom frame is air (ior 1, pri -1, null id).
#include <cstddef>

struct medium_frame {
    double ior = 1.0;
    int pri = -1;
    const void *id = nullptr; // owning prim (host: hittable*; device: id pair)
};

class medium_stack {
public:
    static constexpr int CAP = 8;
    medium_stack() : n(1) { f[0] = medium_frame{}; }

    int depth() const { return n; }
    const medium_frame &top() const { return f[n - 1]; }

    enum event { ENTER, EXIT, PASS };
    // Interface event for a dielectric hit. Updates the stack, reports eta
    // (current_ior / next_ior). PASS = legacy front_face handling, stack
    // untouched (lower-pri surface, same-pri overlap, or overflow past CAP).
    event resolve(double surf_ior, int surf_pri, const void *surf_id, double &eta) {
        const medium_frame &t = f[n - 1];
        if (surf_id != nullptr && surf_id == t.id && n > 1) {
            eta = t.ior / f[n - 2].ior;
            --n;
            return EXIT;
        }
        if (surf_pri > t.pri) {
            eta = t.ior / surf_ior;
            if (n < CAP) {
                f[n] = {surf_ior, surf_pri, surf_id};
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
