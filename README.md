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
src/render/   tile-scheduled multithreaded renderer
src/io/       tonemapping + PPM output
src/main.cpp  scene assembly + bench report
tests/        unit tests (no external framework)
assets/       mesh + calibration textures (copied next to the binary)
```
