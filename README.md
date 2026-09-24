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
  [--split sah|median] [--exposure 1] [--denoise] [--scene default|cornell|weekend|book2]
  [--shutter T0 T1] [--fog D] [--het D] [--aperture A] [--width W] [--height H]
  [--seed 42] [--hdr float.pfm] [--bench] [--noise] [--env]
  [--sampler stratified|sobol] [--aov] [--list-scenes]
  [--maxdepth 50] [--fixed-rng] [--pdf mis|mixture]
```

Renders `out/image.ppm` (400x225, 16spp stratified default,
`--samples 1` reproduces single-sample look). `--hdr` dumps linear
pre-exposure float (PFM) for post-workflows alongside the PPM.
`--noise` adds an opt-in marble showcase sphere (defaults stay frozen).
`--env` enables the analytic sun+sky environment with MIS (off = byte-exact legacy sky).
`--sampler sobol` swaps the pixel set for Cranley-Patterson rotated Sobol-2D (default stratified frozen).
`--aov` dumps linear `out/aov_{albedo,normal,depth}.pfm` for denoise/ML workflows.
`--list-scenes` prints the scene registry (default/cornell/weekend/book2
plus book aliases). weekend/book2 default to 500spp book-style renders.
`--maxdepth` caps bounces (depth-ladder forensics; prod 50). `--fixed-rng`
forces a deterministic 0.5 stream (pixel centers, center light samples)
for exact cross-backend debugging. `--pdf mixture` selects the Book 3
mixture-density path (no shadow rays, full light counts); default `mis`
(power-heuristic MIS) supersedes it on variance (~2.2x on cornell).

```bat
build\Release\rt_gpu.exe [path.spv] [out.ppm] [--spp N] [--chunk C] [--seed S]
  [--scene NAME] [--shutter T0 T1] [--fog D] [--het D] [--width W] [--height H]
  [--hdr float.pfm] [--aperture A] [--exposure X] [--denoise] [--joint]
  [--noise] [--env] [--list-scenes] [--maxdepth 50] [--fixed-rng] [--aov]
  [--pdf mis|mixture]
```

Headless Vulkan compute backend (discrete NVIDIA pick). Statistical CPU
parity by design (wang-hash vs mt19937 RNGs); see Parity below.

`--chunk C` splits spp into TDR-safe dispatches (chunk k uses seed+k)
and averages linear HDR on the host (bit-exact vs the old manual
averaging). Without it, one big dispatch can hit Windows TDR
(`vulkan error -4`) on heavy scenes. Rough 1650 Ti limits at 1200px:
book2 chunk<=2, cornell chunk<=25, default chunk<=50; keep dispatch
under ~2 s. `--denoise`/`--joint` run once on the averaged beauty.
(book2 tightened 6->2 in M59: per-event volume-NEE marches lengthened
dispatches; the table is advisory, watch the dispatch ms.)
`--maxdepth`/`--fixed-rng` mirror the CPU forensics hatches. `--aov`
downloads the albedo/normal guides as PFM (last chunk wins when chunked).
`--pdf mixture` selects the Book 3 path; default `mis`.

```bat
build\Release\rt_view.exe [--scene showcase|default|...] [--gpu|--cpu]
  [--width W] [--height H] [--scale N] [--exposure E]
```

Interactive real-time previewer with 6-DOF fly camera and progressive path tracing:
- **WASD**: Fly forward, backward, strafe left/right
- **Space / C** (or **E / Q**): Fly up / down along world axis
- **Shift / Ctrl**: Sprint (3x speed) / Slow precision sneak (0.25x speed)
- **Right Drag / F**: Look around (pitch & yaw) / Toggle captured mouse look
- **Mouse Wheel**: Adjust movement speed
- **1 - 4**: Instant camera bookmarks (1: Bunny close-up, 2: Crystals, 3: Disney flanks, 4: Wide)
- **F1 - F3**: Live Display Mode (F1: Beauty, F2: Albedo AOV Guide, F3: Normal AOV Guide)
- **X**: Toggle accumulation pause / resume (freeze converged view)
- **G**: Toggle GPU compute vs CPU multi-threading in real time
- **T**: Toggle ACES tonemapping vs standard sRGB gamma
- **[ / ]**: Decrease / Increase exposure multiplier
- **R**: Reset camera to initial scene view
- **P / F12**: Print camera code snippet to console & save snapshot to `out/viewport.ppm`
- **H**: Print full interactive hotkey guide to console
- **Esc**: Release mouse cursor lock / Quit

```bat
build\Release\rt_view.exe [image.ppm] [--diff other.ppm] [--scale N] [--stats]
```

PPM inspector mode: pixel inspector (hover coordinates & RGB), diff heatmap (`D`), file watcher reload on disk change (`R`). `--stats` prints headless diff numbers (mean/max/over8%) without opening a window.

## Porting

Linux/macOS build the same CMake tree (C++20, no Win32 API anywhere).
Per-OS setup:

```sh
# Ubuntu: archive Vulkan packages + SDL3 system deps + Lavapipe
sudo apt-get update -qq && sudo apt-get install -y glslc libvulkan-dev \
  mesa-vulkan-drivers libx11-dev libxext-dev libxrandr-dev libxcursor-dev \
  libxi-dev libxss-dev libxinerama-dev libwayland-dev libxkbcommon-dev \
  libegl1-mesa-dev libpipewire-0.3-dev libpulse-dev libasound2-dev libudev-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build --output-on-failure

# macOS: official SDK installer, then the same three cmake lines.
```

Notes: the app requests Vulkan 1.0 (max compatibility, incl. old
MoltenVK); shaders are SPIR-V 1.0; no Float64 anywhere. Device pick
prefers discrete NVIDIA and degrades to the first compute device
(Lavapipe/Apple Silicon) — the `gpu: <name>` log line names the pick.
CI (`.github/workflows/ci.yml`) builds Windows + Ubuntu, runs unit tests,
and does a tiny Lavapipe `rt_gpu` smoke render on Ubuntu. macOS is out of
the matrix for now: LunarG's mac download is a GUI installer .app (the old
dmg flow is gone), so no headless install path is verified; the tree itself
is portable, re-enable when someone confirms one on Apple hardware.
MoltenVK hardware verification is likewise open (no Apple device here):
timestamps and RGBA32F storage are the two things to eyeball first.

## Parity

Cross-backend diffs are judged against the same-backend different-seed
floor, never an absolute threshold. Method notes from the hunt:

- Compare at matched sampler structure: CPU non-square spp is pure
  random, GPU chunks are stratified grids (CPU-CPU floor 10.4 vs GPU-GPU
  floor 8.3 on book2/500spp is expected sampler variance, not a bug).
  Perfect-square spp stratifies on both; `--sampler sobol` is available
  but changes nothing for cross-backend verdicts.
- Deterministic probes beat blind review: center-ray first-hit t/mtype,
  analytic chord/entry/exit oracles, fixed-point NEE transmittance, event
  rates vs 1-exp(-sL), phase-albedo exactness, fixed-RNG renders (identical
  sample sets both sides). Forensics live in `.scratch/` specs (M48-M54);
  temp probe shaders were reverted pre-commit, kept flags documented above.
- Current standing (1200px/500spp unless noted): default, cornell,
  weekend at floor; fog/het matrix at floor (M48 closed the gap);
  book2 residual CLOSED (M54): depth-1 400px/1024spp cross 1.46 -> 0.60
  vs 0.55 floor; depth-2 400px/256spp cross 2.86 -> 2.44 vs 2.43 floor;
  fixed-RNG d1/d2 cross down 40-100x. Cause: fp32 origin self-skims in
  device shadow + beauty (dense cluster); fix: origin-prim skip inline in
  traversal (shadow unconditional, beauty T-gated 0.5), M51 lift removed.
  Leftover: fp32 silhouette knife-edges (mixed-sign speckle, converges).

### CI Parity Gate & Estimator Drift Harness

CI (`.github/workflows/ci.yml`) runs an automated cross-backend parity gate on Ubuntu using Mesa's Lavapipe software Vulkan driver:
1. **Fixed-RNG Arithmetic Gate**: Runs CPU (`raytracer`) and GPU (`rt_gpu`) with `--fixed-rng` on `default` (64x36, 4 spp). Diffs with `rt_view --diff --stats` and asserts `mean <= 0.05` and `over8 <= 0.1%` (calibrated: `mean ~ 0.002 - 0.004`, `over8 = 0%`). Any arithmetic regressions or shader formula bugs trip this gate instantly without Monte Carlo noise.
2. **Estimator Drift Harness (`tools/parity.py`)**: Tests a matrix across scenes (`default`, `cornell`, `weekend`, `book2`, `sss`) × sampling modes (`mis`, `mixture`) × bounce depths (`1`, `2`, `50`) × RNG modes (`fixed`, `sampled`). Verifies that outputs do not drift from calibrated baselines recorded in `tools/parity_baseline.json`.

**Local debugging & reproduction**:
```bash
# Fast check (default scene, 12 cells, <10s):
ctest -R parity --output-on-failure
# or directly:
python tools/parity.py --fast

# Full 60-cell matrix across all scenes and modes:
python tools/parity.py

# Re-calibrate and establish new baseline:
python tools/parity.py --save-baseline tools/parity_baseline.json
```

## Layout

```text
src/core/      vec3, ray, RNG, sampler (stratified/Sobol), ONB, AABB, textures
               (+mipmaps, noise), OBJ/MTL loader
src/camera/    pinhole + thin-lens defocus + shutter timing
src/geometry/  sphere/triangle/quad, hittable list, constant + hetero volumes,
               instances (translate/rotate_y)
src/material/  lambertian/metal/dielectric/isotropic/diffuse_light, GGX
               (iso + aniso conductors, rough glass), Ns mapping
src/integrator/ NEE + MIS path integrator (power heuristic, transmittance
               weighting), first-hit AOV guides
src/accel/     median + binned-SAH BVH, QBVH-4 collapse (SSE2 slabs), flat QBVH upload twin
src/scene/     per-scene modules (common/default/cornell/weekend/book2) +
               registry, shared CPU/GPU construction order
src/gpu/       Vulkan compute host (chunked submit, HDR average), flatten
               (instances bake, black_bg), shaders (grad/normal/path/
               denoise/joint), CPU-mirrored camera
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
M39 done: GPU QBVH-4 traversal (flat collapse twin, bit-exact CPU mirror), binary path deleted.
M40 done: GPU `--aperture` thin-lens + `--exposure` film parity (pinhole streams bit-stable).
M41 done: rotated Sobol-2D pixel sampler (`--sampler sobol`) + linear AOV trio dump (`--aov`).
M42 done: noise audit (remote value noise kept, marble parity at floor).
M43 done: Walter rough-glass BTDF for dielectrics.
M44 done: NEE transmittance weighting in volumes (gap closed in M48).
M45 done: GPU joint-bilateral denoise with on-device guides.
M46 done: MTL Ns to GGX roughness mapping.
M47 done: anisotropic GGX conductors with UV tangents, brushed-metal demo.
M48 done: fog parity hunt closed (GPU scatter-continuation + CPU
shadow-march exit-advance + hit_obj staleness); fog/het matrix at floor.
M49 done: book2 re-measure (23.4 -> 15.2); sampler-variance analysis;
residual decomposed, documented.
M50 done: device probes (fog geometry/sampling/upload exonerated exactly).
M51 done: book2 residual work (traverse fog-skip for co-located shells +
shadow bias for fp32 self-skims; 15.2 -> 11.5 vs ~10.4 floor).
M52 done: GPU `--chunk` in-binary HDR averaging (bit-exact vs manual).
M53 done: default showcase tune (brushed ball, key 6, studio void,
vfov 75) + GPU camera mirrored from CPU (hardcoded drift fixed).
M54 done: book2 residual closed (origin-prim skip inline in device
traversal: shadow unconditional, beauty T-gated 0.5; M51 lift removed;
NEE range mirrors CPU). Depth-1/d2 cross at floor; fixed-RNG down 40-100x.
GPU showcases re-anchored (VII); CPU untouched. Kept forensics flags:
--maxdepth/--fixed-rng (both), GPU --aov readback.
M55 done: port prep (Vulkan 1.0 floor, glslc bin/ hints, .gitattributes)
+ CI (Windows + Ubuntu build/test, Lavapipe smoke; macOS out: LunarG mac
download is a GUI .app now, no verified headless install).
M56 done: full-depth 1200/500 proof (CPU hash == VI anchor; cross 3.31 vs
2.97 floor, signed ~0; leftover = symmetric fp32-geometry noise floor).
M59 done: NEE at volume events (direct in-scattering, MIS vs 1/4PI phase;
fixed-RNG exact; sss floor -17%, milk glows; book2/cross at floor;
default/cornell/weekend bit-identical pre/post). Anchors VIII (estimator
changed volume scenes). TDR table: book2 chunk<=2.
M60 done: density-gated volume NEE (fire >= 0.01 majorant: haze skips
back to walks, milk/smoke fire; unbiased, sample-independent gate).
book2 d1k floor recovered to 0.55, full cross 3.56 vs 2.98 floor signed
~0; sss bit-identical to M59. Anchors IX (book2 only).
M61 done: Book 3 mixture-density path as `--pdf mixture` alternate (both
backends; no shadow rays, full light counts; fixed-RNG exact; cornell
converges to MIS mean at 2.2x the noise). Default stays power-MIS.
M63 done: HDRI environment map importance sampling (separable 2D luminance CDF,
stb_image HDR load, MIS-correct miss + env NEE for surface & volume, GPU bindings 12/13).
M67 done: spectral hero transport (hero-wavelength pick per path, Cauchy dispersion,
exact conductor Fresnel from measured n/k, 3xLi compensation; GPU mirrors CPU).
M68 done: physical thin-film interference (Airy equations, multi-layer reflection
for metals and dielectrics, iridescent soap bubble and oxide sheen).
M69 done: correctness deepening & layout audit (alb2 layout audit, GPU glass delta +
rough thin-film Fresnel fix, unit tests).
M70 done: Disney Principled BRDF (retro-reflective diffuse, GGX VNDF specular,
grazing sheen, GTR1 clearcoat, subsurface) on CPU & GPU pipelines with 15,630 checks;
integrated Midnight Sapphire Car Paint and Royal Crimson Velvet into showcase masterpiece.
M71 done: interactive real-time previewer in rt_view with 6-DOF fly-camera (WASD +
mouse look), progressive path tracing with instant accumulation reset on motion, and
real-time GPU/CPU dual-engine switching.
M72 done: smooth mesh tangent continuity & geometric normal consistency clamping
(continuous ONB derivation for smooth non-UV meshes; hemispherical reflection clamping
preventing specular rays from tunneling into hollow meshes; camera bookmarks in rt_view).
M73 done: tangent-space normal mapping & Disney pearlescent clearcoat (Option 2 & Option 3).
Tangent-space normal mapping perturbing shading frames via TBN matrix on Lambertian,
Conductor, and Disney BSDFs; procedural wave normal texture and OBJ MTL map_Bump/bump/norm
parser support; first_hit_aov normal guide alignment. Iridescent pearlescent clearcoat
layering physical thin-film interference (Airy equations) onto Disney GTR1 clearcoat
on CPU & GPU with zero buffer layout disruption; integrated into Midnight Sapphire Car Paint;
15,639 checks passed, 100% CPU/GPU parity.
Trilogy mapping complete: every chapter implemented or superseded.
Next: MoltenVK verify (blocked, no hardware).

