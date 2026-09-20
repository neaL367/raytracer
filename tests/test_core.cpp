// Core math/RNG/sampling tests: vec3, ray, sampler, Monte Carlo, ONB, RR.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "core/onb.h"

static void t_vec3() {
    test_current = "vec3";
    vec3 a(1, 2, 3), b(4, -1, 0.5);
    vec3 s = a + b;
    EXPECT_NEAR(s.x(), 5); EXPECT_NEAR(s.y(), 1); EXPECT_NEAR(s.z(), 3.5);
    EXPECT_NEAR(dot(a, b), 3.5);
    EXPECT_NEAR(cross(vec3(1,0,0), vec3(0,1,0)).z(), 1);
    EXPECT_NEAR(unit_vector(vec3(0,3,4)).length(), 1);
    EXPECT_NEAR(reflect(vec3(1,-1,0), vec3(0,1,0)).y(), 1); // mirror
}

static void t_ray() {
    test_current = "ray";
    EXPECT_NEAR(ray(vec3(0,0,0), vec3(0,0,-1)).at(0.5).z(), -0.5);
}

static void t_sampler() {
    test_current = "sampler";
    rng_seed(10);
    auto j = jitter_offsets(50);
    EXPECT_TRUE((int)j.size() == 50);
    for (auto [ox, oy] : j)
        EXPECT_TRUE(ox >= 0 && ox < 1 && oy >= 0 && oy < 1);
    rng_seed(10);
    auto s = stratified_offsets(4); // 16 samples, 4x4 cells
    EXPECT_TRUE((int)s.size() == 16);
    int cells[4][4] = {};
    for (auto [ox, oy] : s) {
        EXPECT_TRUE(ox >= 0 && ox < 1 && oy >= 0 && oy < 1);
        int cx = (int)(ox * 4), cy = (int)(oy * 4);
        if (cx > 3) cx = 3;
        if (cy > 3) cy = 3;
        cells[cy][cx]++;
    }
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            EXPECT_TRUE(cells[y][x] == 1); // each cell exactly once
    auto one = pixel_samples(1);
    EXPECT_TRUE(one.size() == 1 && test_near(one[0].first, 0.5) &&
                test_near(one[0].second, 0.5));
    EXPECT_TRUE((int)pixel_samples(7).size() == 7); // non-square -> jitter
}

// Monte Carlo: integrate f(x,y)=x+y over unit square (truth=1.0).
// 20 seeded trials each; stratified variance must beat jitter.
static double trial_estimate(bool stratified, unsigned seed) {
    rng_seed(seed);
    double sum = 0;
    if (stratified) {
        for (auto [ox, oy] : stratified_offsets(4))
            sum += ox + oy;
        return sum / 16;
    }
    for (auto [ox, oy] : jitter_offsets(16))
        sum += ox + oy;
    return sum / 16;
}

static void t_montecarlo() {
    test_current = "montecarlo";
    double ms = 0, mj = 0;
    double vs = 0, vj = 0;
    const int T = 20;
    double es[T], ej[T];
    for (int t = 0; t < T; ++t) {
        es[t] = trial_estimate(true, 100 + (unsigned)t);
        ej[t] = trial_estimate(false, 200 + (unsigned)t);
        ms += es[t]; mj += ej[t];
    }
    ms /= T; mj /= T;
    for (int t = 0; t < T; ++t) {
        vs += (es[t] - ms) * (es[t] - ms);
        vj += (ej[t] - mj) * (ej[t] - mj);
    }
    vs /= T; vj /= T;
    EXPECT_TRUE(fabs(ms - 1.0) < 0.05); // converges to truth
    EXPECT_TRUE(fabs(mj - 1.0) < 0.15);
    EXPECT_TRUE(vs < vj); // strata win on variance
}

static void t_onb_cosine() {
    test_current = "onb";
    onb frame;
    frame.build_from_w(vec3(0, 1, 0));
    EXPECT_NEAR(frame.u.length(), 1);
    EXPECT_NEAR(frame.v.length(), 1);
    EXPECT_NEAR(frame.w.length(), 1);
    EXPECT_NEAR(dot(frame.u, frame.v), 0);
    EXPECT_NEAR(dot(frame.v, frame.w), 0);
    EXPECT_NEAR(dot(frame.w, frame.u), 0);
    rng_seed(20);
    double mean_z = 0;
    const int N = 10000;
    for (int i = 0; i < N; ++i)
        mean_z += random_cosine_direction().z();
    mean_z /= N;
    EXPECT_TRUE(fabs(mean_z - 0.6667) < 0.02); // pdf z/PI integrates so
    EXPECT_NEAR(cosine_pdf(1.0), 1.0 / 3.1415926535897932385);
    EXPECT_NEAR(cosine_pdf(-0.5), 0.0);
}

static void t_rr() {
    test_current = "rr";
    rng_seed(24);
    // Mirror of integrator RR: q=0.5 fixed throughput, mean preserved.
    double sum = 0;
    const int T = 20000;
    for (int i = 0; i < T; ++i) {
        double tput = 0.5;
        double q = tput < 0.95 ? tput : 0.95;
        double v = (random_double() > q) ? 0.0 : tput / q;
        sum += v;
    }
    EXPECT_TRUE(fabs(sum / T - 0.5) < 0.02); // unbiased termination
}

void run_core_tests() {
    t_vec3();
    t_ray();
    t_sampler();
    t_montecarlo();
    t_onb_cosine();
    t_rr();
}
