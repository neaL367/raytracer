#pragma once
// Scene dispatcher: showcase and default scenes.
// main() only wires (BVH + render); placement lives in the per-scene headers.
#include "common.h"
#include "showcase.h"
#include "default.h"

inline scene_data build_scene(const std::string &name, double aspect, double aperture,
                                double sh0 = 0, double sh1 = 0, double fog = 0,
                                double het = 0, bool marble = false, bool env = false) {
    if (name == "showcase" || name == "show" || name == "demo")
        return build_showcase(aspect, aperture, sh0, sh1, fog, het, marble, env);
    return build_default(aspect, aperture, sh0, sh1, fog, het, marble, env);
}
