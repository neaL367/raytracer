// Test runner: thin main only. Suites live in test_core/geometry/
// shading/scene TUs behind run_* seams in test_helpers.h.
// Dependency-free: no Catch2, deterministic seeded math checks.
#include "test_helpers.h"

#include <cstdio>

int main() {
    run_core_tests();
    run_geometry_tests();
    run_shading_tests();
    run_scene_tests();
    std::printf("%d checks, %d failures\n", test_checks, test_failures);
    return test_failures ? 1 : 0;
}
