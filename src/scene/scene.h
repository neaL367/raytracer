#pragma once
// Scene dispatcher: one builder per showcase scene, stable names.
// main() only wires (BVH + render); all placement lives in the per-scene
// headers (common.h holds shared plumbing: scene_data, box helpers).
#include "common.h"
#include "showcase.h"
#include "default.h"
#include "cornell.h"
#include "weekend.h"
#include "book2.h"
#include "sss.h"

inline scene_data build_scene(const std::string &name, double aspect, double aperture,
                                double sh0 = 0, double sh1 = 0, double fog = 0,
                                double het = 0, bool marble = false, bool env = false) {
    if (name == "cornell")
        return build_cornell(aspect, aperture, env);
    if (name == "weekend" || name == "final" || name == "spheres" || name == "book1")
        return build_weekend(aspect, aperture, sh0, sh1, env);
    if (name == "book2" || name == "nextweek" || name == "boxes" || name == "final2")
        return build_book2(aspect, aperture, sh0, sh1, env);
    if (name == "sss")
        return build_sss(aspect, aperture);
    if (name == "showcase" || name == "show" || name == "demo")
        return build_showcase(aspect, aperture, sh0, sh1, fog, het, marble, env);
    return build_default(aspect, aperture, sh0, sh1, fog, het, marble, env);
}
