# raytracer

CPU path tracer from scratch, C++20, MSVC 2022. Target: GTX 1650 Ti class.

## Build

```bat
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

## Run

```bat
build\Release\raytracer.exe [--samples 16] [--threads N] [--bench]
```

Renders `out/image.ppm` (400x225, 16spp stratified default,
`--samples 1` reproduces single-sample look).

## Layout

```text
src/core/      vec3, ray (value types, hot loop)
src/camera/    pinhole ray generation
src/geometry/  sphere hit interface
src/output/    PPM writer (linear film, gamma at write)
src/app/       wiring, render loop
tests/         dependency-free asserts via ctest
.scratch/      specs + tickets (local, gitignored)
```

## Roadmap

M0/M1 done: toolchain + first sphere image.
M2 done: geometry + materials + lights.
M3 done: stratified sampling + AA.
M4 done: NEE + MIS path tracing, RR, defocus.
M5 done: median BVH, tile thread pool, bench counters.
M6 done: headless Vulkan compute backend (rt_gpu), CPU parity.
M7 done: textures + OBJ meshes, checker demo, GPU parity.
M8 done: exposure + ACES + sRGB film, shared CPU/GPU writer.
M9 done: binned SAH BVH, --split flag, prim tests halved.
M10 done: HDR bilateral --denoise, 1.25x vs 64spp ref.
M11 done: Cornell box, scene module, camera focus fix.
M12 done: guided joint-bilateral denoise, 6.57x vs ref.
M13 done: GPU BVH traversal, parity 0.11%, brute deleted.
M14 done: stb JPEG/PNG, photo ground, GPU image path.
M15 done: bilinear sampling both backends, parity 0.6%.
M16 done: OBJ smooth normals, zero pixel drift.
M17 done: multi-image registry, two-photo demo, parity 0.2%.
M18 done: mesh UV mapping, photo-per-face cube, parity 0.2%.
M19 done: motion blur via shutter + moving spheres, defaults frozen.
M20 done: constant-density fog volumes, opt-in glow.
M21 done: GPU motion + fog parity, path-time inheritance.
Next: pick slice.
