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
build\Release\raytracer.exe [--samples 16] [--threads N] [--tile 8]
  [--split sah|median] [--exposure 1] [--denoise] [--scene default|cornell]
  [--shutter T0 T1] [--fog D] [--aperture A] [--width W] [--height H]
  [--seed 42] [--hdr float.pfm] [--bench] [--noise] [--env]
```

Renders `out/image.ppm` (400x225, 16spp stratified default,
`--samples 1` reproduces single-sample look). `--hdr` dumps linear
pre-exposure float (PFM) for post-workflows alongside the PPM.
`--noise` adds an opt-in marble showcase sphere (defaults stay frozen).
`--env` enables the analytic sun+sky environment with MIS (off = byte-exact legacy sky).

```bat
build\Release\rt_gpu.exe [path.spv] [out.ppm] [--spp N] [--seed S]
  [--scene NAME] [--shutter T0 T1] [--fog D] [--width W] [--height H]
  [--hdr float.pfm]
```

Headless Vulkan compute backend (discrete NVIDIA pick). Statistical CPU
parity by design (wang-hash vs mt19937 RNGs); fog parity is judged against
the same-backend different-seed floor, never an absolute threshold.

```bat
build\Release\rt_view.exe [image.ppm] [--diff other.ppm] [--scale N] [--stats]
```

SDL3 preview: pixel inspector, diff heatmap (`D`), watcher reload (`R`).
`--stats` prints headless diff numbers (mean/max/over8%).

## Layout

```text
src/core/      vec3, ray, RNG, sampler, ONB, AABB, textures (+mipmaps, noise),
               OBJ/MTL loader
src/camera/    pinhole + thin-lens defocus + shutter timing
src/geometry/  sphere/triangle/quad, hittable list, constant-density fog
src/material/  lambertian/metal/dielectric/isotropic/diffuse_light
src/integrator/ NEE + MIS path integrator, first-hit AOV guides
src/accel/     median + binned-SAH BVH, QBVH-4 collapse (SSE2 slabs), flatten accessors
src/scene/     default + cornell builders (shared CPU/GPU construction order)
src/gpu/       Vulkan compute host, flatten, shaders (grad/normal/path)
src/output/    PPM writer, ACES film, PFM float dump
src/io/        denoise (bilateral + joint), compare/heatmap, PPM + stb images
src/app/       CPU wiring, tile thread pool, bench counters
src/view/      SDL3 preview + inspector
tests/         dependency-free asserts via ctest (seeded, deterministic)
.scratch/      specs + tickets + roadmap (local, gitignored)
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
M22 done: SDL3 preview + pixel inspector + diff mode.
M23 done: CPU `--seed`, fog parity verdict (statistical, floor rule).
M24 done: mipmapped textures with distance LOD + per-texture span.
M25 done: MTL materials (Kd/Ks/map_Kd) per-face in OBJ loader.
M26 done: PFM float dump (`--hdr`) on both backends + doc refresh.
M27 done: GGX conductors with VNDF sampling (fuzz replaced, mtype 7).
M28 done: NEE for sphere + mesh lights, warm orb demo light.
M29 done: per-vertex mesh motion blur, cube drifts with shutter.
M30 done: GPU bilateral denoise post-pass (`rt_gpu --denoise`).
M31 done: heterogeneous volumes with delta tracking (two GPU volume bugs fixed).
M32 done: CPU QBVH-4 collapse with SSE2 slabs, bit-exact traversal.
M33 done: instance module (translate + rotate_y + posed boxes), rotated Cornell blocks, GPU bake parity.
M34 done: Perlin value-noise + fBm + turbulence + marble textures, opt-in `--noise` demo, type-9 GPU path.
M35 done: direction-PDF module (cosine + NEE + 50/50 mixture) beside the integrator, zero pixel change.
M36 done: power-heuristic MIS on CPU, first-class direction densities, isotropic found-lights MIS-weighted.
M37 done: GPU power-heuristic MIS parity in path shader (found-light + NEE weights mirrored).
M38 done: analytic sun+sky environment with uniform-sphere NEE + MIS on both backends, `--env` opt-in.
Next: `.scratch/roadmap.md` backlog (Linux/macOS port).
