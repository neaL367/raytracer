// Shared test harness: counters, check(), near(), EXPECT macros.
// Single TU owns nothing; inline variables keep one state across TUs.
#pragma once

#include <cmath>
#include <cstdio>

inline int test_checks = 0;
inline int test_failures = 0;
inline const char *test_current = "";

inline void test_check(bool cond, const char *expr, int line) {
    ++test_checks;
    if (!cond) {
        ++test_failures;
        std::printf("  FAIL %s:%d: %s\n", test_current, line, expr);
    }
}

inline bool test_near(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

#define EXPECT_TRUE(x) test_check((x), #x, __LINE__)
#define EXPECT_NEAR(a, b) test_check(test_near((a), (b)), #a " ~= " #b, __LINE__)

void run_core_tests();
void run_geometry_tests();
void run_shading_tests();
void run_scene_tests();
