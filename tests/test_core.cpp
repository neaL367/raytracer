// Core math/RNG/sampling tests: vec3, ray, sampler, Monte Carlo, ONB, RR.
#include "test_helpers.h"
#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "core/onb.h"
#include "core/spectrum.h"

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

static void t_sobol() {
    test_current = "sobol";
    // Gray-order goldens: i=1 (1/2,1/2), i=2 (3/4,1/4), i=3 (1/4,3/4).
    EXPECT_NEAR(sobol_dim0(0), 0.0);
    EXPECT_NEAR(sobol_dim1(0), 0.0);
    EXPECT_NEAR(sobol_dim0(1), 0.5);
    EXPECT_NEAR(sobol_dim1(1), 0.5);
    EXPECT_NEAR(sobol_dim0(2), 0.75);
    EXPECT_NEAR(sobol_dim1(2), 0.25);
    EXPECT_NEAR(sobol_dim0(3), 0.25);
    EXPECT_NEAR(sobol_dim1(3), 0.75);
    // (0,2)-sequence: first 16 points tile every 4x4 cell exactly once.
    int cells[4][4] = {};
    for (unsigned k = 0; k < 16; ++k) {
        int cx = (int)(sobol_dim0(k) * 4), cy = (int)(sobol_dim1(k) * 4);
        if (cx > 3)
            cx = 3;
        if (cy > 3)
            cy = 3;
        cells[cy][cx]++;
    }
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            EXPECT_TRUE(cells[y][x] == 1);
    // Shifted set stays in-bounds and keeps cardinality.
    rng_seed(11);
    auto offs = sobol_offsets(16);
    EXPECT_TRUE((int)offs.size() == 16);
    for (auto [ox, oy] : offs)
        EXPECT_TRUE(ox >= 0 && ox < 1 && oy >= 0 && oy < 1);
    // Smooth integrand: Sobol beats same-count jitter on this seed.
    auto fintegrand = [](const std::vector<sample_offset> &pts) {
        double s = 0;
        for (auto [x, y] : pts)
            s += x * x + y * y; // truth over [0,1]^2 = 2/3
        return s / pts.size();
    };
    rng_seed(12);
    double es = fabs(fintegrand(sobol_offsets(64)) - 2.0 / 3.0);
    rng_seed(12);
    double ej = fabs(fintegrand(jitter_offsets(64)) - 2.0 / 3.0);
    EXPECT_TRUE(es < 0.02); // low-discrepancy converges fast
    EXPECT_TRUE(es <= ej);
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

static void t_rr() {    test_current = "rr";
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

static void t_seed_streams() {
    test_current = "seed_streams";
    // --seed contract: same seed replays, neighbors decorrelate.
    rng_seed(42);
    double a1 = random_double(), a2 = random_double();
    rng_seed(42);
    EXPECT_TRUE(random_double() == a1 && random_double() == a2); // replay
    rng_seed(43);
    double b1 = random_double();
    EXPECT_TRUE(b1 != a1); // adjacent pixel stream differs
}

static void t_spectrum() {
    test_current = "spectrum";
    // Hero wavelengths distinct, R/G/B ordered.
    EXPECT_TRUE(spectrum::kHeroLambda[0] == 650.0);
    EXPECT_TRUE(spectrum::kHeroLambda[1] == 550.0);
    EXPECT_TRUE(spectrum::kHeroLambda[2] == 450.0);
    // pick(): off = identity; on = hero channel value, 0 elsewhere.
    // Weight-free: E[pick] over heroes = v/3, restored by the single x3
    // at the end of Li (per-quantity weights would compound to 9x).
    vec3 v(1, 2, 3);
    vec3 id = spectrum::pick(v, 1, false);
    EXPECT_TRUE(id.x() == 1 && id.y() == 2 && id.z() == 3);
    vec3 pk = spectrum::pick(v, 1, true);
    EXPECT_TRUE(pk.x() == 0 && pk.y() == 2 && pk.z() == 0);
    vec3 mean(0, 0, 0);
    for (int c = 0; c < 3; ++c)
        mean = mean + spectrum::pick(v, c, true);
    // Each channel picked exactly once across heroes: sum == v, and the
    // single x3 in Li compensates the 1/3 sampling probability.
    EXPECT_TRUE(mean.x() == 1 && mean.y() == 2 && mean.z() == 3);
    // Cauchy: B=0 identity; B>0 normal dispersion (blue bends more).
    EXPECT_NEAR(spectrum::cauchy_ior(1.5, 0.0, 450.0), 1.5);
    double nred = spectrum::cauchy_ior(1.52, 0.0042, 650.0);
    double nblue = spectrum::cauchy_ior(1.52, 0.0042, 450.0);
    EXPECT_TRUE(nblue > 1.52 && 1.52 > nred); // d-line ref between red/blue
    // Hand check: l=0.65um, lr=0.5876um.
    double hand = 1.52 + 0.0042 * (1.0 / (0.65 * 0.65) - 1.0 / (0.5876 * 0.5876));
    EXPECT_TRUE(fabs(nred - hand) < 1e-12);
    // Exact conductor Fresnel: k=0 dielectric limit R0=((n-1)/(n+1))^2.
    double r0 = spectrum::conductor_R(1.0, 1.5, 0.0, 1.0);
    EXPECT_TRUE(fabs(r0 - 0.04) < 1e-9);
    // Grazing -> 1, energy bounded, normal-incidence gold is reddish.
    EXPECT_TRUE(spectrum::conductor_R(1.0, 0.35, 2.75, 0.0) > 0.99);
    double nau, kau;
    EXPECT_TRUE(spectrum::conductor_nk(1, 0, nau, kau)); // Au @650
    double nau_b, kau_b;
    EXPECT_TRUE(spectrum::conductor_nk(1, 2, nau_b, kau_b)); // Au @450
    double r_red = spectrum::conductor_R(1.0, nau, kau, 1.0);
    double r_blue = spectrum::conductor_R(1.0, nau_b, kau_b, 1.0);
    EXPECT_TRUE(r_red > 0.9 && r_blue < r_red); // gold reflects red, eats blue
    EXPECT_TRUE(!spectrum::conductor_nk(0, 1, nau, kau)); // preset 0 = none
    EXPECT_TRUE(!spectrum::conductor_nk(5, 1, nau, kau));
    EXPECT_TRUE(!spectrum::conductor_nk(1, 3, nau, kau));
    // Amplitude pair consistent with intensity version.
    std::complex<double> rs, rp;
    spectrum::fresnel_ri(1.0, 0.0, nau, kau, 1.0, rs, rp);
    double ramp = 0.5 * (std::norm(rs) + std::norm(rp));
    EXPECT_TRUE(fabs(ramp - r_red) < 1e-9);
}

void run_core_tests() {
    t_vec3();
    t_ray();
    t_sampler();
    t_sobol();
    t_montecarlo();
    t_onb_cosine();
    t_rr();
    t_seed_streams();
    t_spectrum();
}
