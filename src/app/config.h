#pragma once
// All command-line configuration in one struct. parse_cli fills it;
// main.cpp only reads it. Derived values (stratified) are computed here so
// every consumer sees consistent settings.
#include <cmath>
#include <iostream>
#include <string>

struct render_config
{
    bool bench = false;
    int extra_spheres = 300;
    int samples_per_pixel = 196;
    int max_depth = 50;
    int tile_rows = 8;
    double aperture = 0.05;
    unsigned bench_seed = 42u;
    size_t max_leaf_size = 2;
    bool ground_in_bvh = true;
    bool do_nee = false;
    bool do_rr = true;
    bool do_strat = true;
    bool do_bilinear = true;
    // "full" path tracing or "normal" reference shading (closest-hit
    // normals, no materials/lights/RNG) for backend parity checks.
    std::string shade_mode = "full";
    double exposure = 1.0;
    // Stratified pixel sampling needs a perfect square count; anything else
    // falls back to plain jitter.
    int strat_n = 1;
    bool stratified = false;
};

inline void print_usage(const char *prog)
{
    std::cout << "usage: " << prog << " [options]\n"
              << "  --bench            timing/counter report with fixed RNG seed\n"
              << "  --spheres N        added random spheres (default 300)\n"
              << "  --samples N        samples per pixel (default 196 = 14x14 strata;\n"
              << "                     must be a perfect square for stratification)\n"
              << "  --shade MODE       full path tracing (default) or normal reference\n"
              << "                     shading for backend parity checks\n"
              << "  --depth N          max path depth (default 50)\n"
              << "  --tile N           scheduling strip height in rows (default 8)\n"
              << "  --aperture A       lens aperture (default 0.05; 0 = pinhole)\n"
              << "  --leaf N           BVH leaf capacity (default 2)\n"
              << "  --ground outside   test the ground sphere separately from the tree\n"
              << "  --nee              next-event estimation against an area light\n"
              << "  --norr             disable russian roulette\n"
              << "  --nostrat          disable stratified sampling (pure jitter)\n"
              << "  --noblinear        nearest-texel image sampling\n"
              << "  --exposure E       linear HDR scale before tonemapping (default 1.0)\n"
              << "  --seed N           fixed RNG seed (default 42)\n"
              << "  --help             this text\n";
}

enum class cli_result
{
    run,  // config filled, render
    help, // --help printed, exit 0 without rendering
    error // usage error printed, exit 1
};

inline cli_result parse_cli(int argc, char **argv, render_config &cfg)
{
    // Value following a flag at argv[i]; advances i past it. ok=false when
    // the flag is the last argument.
    auto take_value = [&](int &i, const char *flag, bool &ok) -> std::string {
        if (i + 1 >= argc)
        {
            std::cerr << flag << " needs a value\n";
            ok = false;
            return "";
        }
        ok = true;
        return argv[++i];
    };

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            bool ok = true;
            if (arg == "--bench")
                cfg.bench = true;
            else if (arg == "--spheres")
                cfg.extra_spheres = std::stoi(take_value(i, "--spheres", ok));
            else if (arg == "--samples")
                cfg.samples_per_pixel = std::stoi(take_value(i, "--samples", ok));
            else if (arg == "--depth")
                cfg.max_depth = std::stoi(take_value(i, "--depth", ok));
            else if (arg == "--tile")
                cfg.tile_rows = std::stoi(take_value(i, "--tile", ok));
            else if (arg == "--aperture")
                cfg.aperture = std::stod(take_value(i, "--aperture", ok));
            else if (arg == "--leaf")
                cfg.max_leaf_size = static_cast<size_t>(std::stoul(take_value(i, "--leaf", ok)));
            else if (arg == "--ground")
                cfg.ground_in_bvh = (take_value(i, "--ground", ok) != "outside");
            else if (arg == "--nee")
                cfg.do_nee = true;
            else if (arg == "--norr")
                cfg.do_rr = false;
            else if (arg == "--nostrat")
                cfg.do_strat = false;
            else if (arg == "--noblinear")
                cfg.do_bilinear = false;
            else if (arg == "--shade")
                cfg.shade_mode = take_value(i, "--shade", ok);
            else if (arg == "--exposure")
                cfg.exposure = std::stod(take_value(i, "--exposure", ok));
            else if (arg == "--seed")
                cfg.bench_seed = static_cast<unsigned>(std::stoul(take_value(i, "--seed", ok)));
            else if (arg == "--help")
            {
                print_usage(argv[0]);
                return cli_result::help;
            }
            else
            {
                std::cerr << "unknown argument: " << arg << "\n";
                print_usage(argv[0]);
                return cli_result::error;
            }
            if (!ok)
                return cli_result::error;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "bad argument value: " << e.what() << "\n";
        return cli_result::error;
    }

    cfg.strat_n = static_cast<int>(std::sqrt(cfg.samples_per_pixel + 0.5));
    cfg.stratified = cfg.do_strat && cfg.strat_n * cfg.strat_n == cfg.samples_per_pixel && cfg.strat_n > 0;
    if (cfg.shade_mode != "full" && cfg.shade_mode != "normal")
    {
        std::cerr << "--shade must be full or normal\n";
        return cli_result::error;
    }
    if (cfg.bench && cfg.do_strat && !cfg.stratified)
        std::cout << "[bench] samples=" << cfg.samples_per_pixel << " not square: jitter fallback\n";
    return cli_result::run;
}
