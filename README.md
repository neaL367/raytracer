# raytracer

CPU path tracer in C++20: BVH (binned SAH) + multithreaded tile rendering,
with a growing light-transport core (area lights, NEE + MIS, russian
roulette, stratified sampling, textures, ACES output).

## Build

Requires CMake ≥ 3.25 and MSVC 2022 (or any C++20 compiler).

```bat
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Tests (dependency-free, run with CTest):

```bat
ctest --test-dir build -C Release
```

## Run

```bat
build\Release\raytracer.exe --bench
```

Renders `output.ppm` (800×450, 196 samples/px by default).
`raytracer.exe --help` lists all flags.

## Deterministic benchmarking

`--bench` fixes the RNG seed, so identical flags produce a bit-identical
image: the reported `image_hash` is the regression test. Any slice that
changes sampling or shading re-anchors the hash; pure refactors must not
move it. Per-thread timings, BVH census, and box/primitive counters come
from the same run.

## Layout

```text
src/core/     math + scene primitives (vec3, ray, shapes, BVH, materials)
src/app/      CLI configuration
src/scene/    scene assembly shared by CPU and GPU backends
src/render/   tile-scheduled multithreaded renderer
src/io/       tonemapping + PPM output + image decoding glue
src/gpu/      headless Vulkan compute (rt_gpu) + GLSL kernels
src/main.cpp  scene/BVH setup + bench report
external/stb/ vendored stb_image v2.30 (PNG/JPG/BMP/TGA decoding)
Intel OIDN is downloaded at configure time into the build tree (never
committed); -DRAYTRACER_OIDN=OFF builds fully offline with a bilateral
fallback for --oidn.
tests/        unit tests (no external framework) + PPM compare script
assets/       mesh + calibration textures (copied next to the binary)
```

GPU backends (`rt_gpu fill|normal|path [samples] [flat|bvh] [spheres] [glass] [WxH] [fog] [aperture]`,
`rt_gpu probe [samples] [spheres]`, `rt_gpu oidn color.ppm albedo.ppm normal.ppm out.ppm`):
`normal` must match the CPU `--shade normal` reference within `tests/compare_ppm.py`
tolerances (fp32 vs fp64); `path` holds statistical parity (means within
a few percent) against `--bench --nee`.
The device traverses the same SAH tree (flattened upload); `flat` keeps
the brute-force loop for A/B. Measured: tree and loop agree bit-exactly
at 316 prims (brute force wins SIMT there), 4.5x tree win at 3000.
Every dispatch reports device-side milliseconds (`[gpu] dispatch=`) via
timestamp queries — wall clock includes scene build, upload, and PPM
write, so the dispatch line is the honest number (e.g. 412ms device vs
26s CPU render on the 196spp bench scene).
`probe` prints a per-pixel path-length histogram: mean 1.64 segments at
64spp with 93% of pixels at ~1 — too tight a distribution for wavefront
compaction to pay off (see commit message for the full verdict).
