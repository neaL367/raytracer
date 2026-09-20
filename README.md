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
build\Release\raytracer.exe
```

Renders `out/image.ppm` (400x225, normal-shaded sphere).

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
Next: M2 geometry + materials + lights.
