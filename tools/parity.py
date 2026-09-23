#!/usr/bin/env python3
"""
Estimator drift harness (M54/M62 parity verification).
Runs a matrix of scenes x sampling modes x bounce depths x RNG modes (fixed-RNG vs sampled)
across CPU (raytracer) and GPU (rt_gpu), diffing the resulting images via rt_view --stats.

Usage:
    python tools/parity.py --fast
    python tools/parity.py --save-baseline tools/parity_baseline.json
    python tools/parity.py --compare-baseline tools/parity_baseline.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def find_binary(names, search_dirs):
    for d in search_dirs:
        for name in names:
            p = Path(d) / name
            if p.is_file():
                return str(p.resolve())
            p_exe = Path(d) / f"{name}.exe"
            if p_exe.is_file():
                return str(p_exe.resolve())
    return None


def run_cell(cpu_bin, gpu_bin, spv_bin, view_bin, scene, mode, depth, rng_mode,
             spp, w, h, out_dir):
    cell_id = f"{scene}_{mode}_d{depth}_{rng_mode}"
    cpu_ppm = str(Path(out_dir) / f"cpu_{cell_id}.ppm")
    gpu_ppm = str(Path(out_dir) / f"gpu_{cell_id}.ppm")
    is_fixed = (rng_mode == "fixed")

    # CPU command
    cpu_cmd = [
        cpu_bin,
        "--scene", scene,
        "--width", str(w),
        "--height", str(h),
        "--spp", str(spp),
        "--pdf", mode,
        "--maxdepth", str(depth),
        cpu_ppm,
    ]
    if is_fixed:
        cpu_cmd.append("--fixed-rng")

    # GPU command
    gpu_cmd = [
        gpu_bin,
        spv_bin,
        "--scene", scene,
        "--width", str(w),
        "--height", str(h),
        "--spp", str(spp),
        "--pdf", mode,
        "--maxdepth", str(depth),
        gpu_ppm,
    ]
    if is_fixed:
        gpu_cmd.append("--fixed-rng")

    t0 = time.perf_counter()
    res_cpu = subprocess.run(cpu_cmd, capture_output=True, text=True)
    t_cpu = time.perf_counter() - t0
    if res_cpu.returncode != 0:
        raise RuntimeError(f"CPU render failed for {cell_id}:\n{res_cpu.stderr}\n{res_cpu.stdout}")

    t0 = time.perf_counter()
    res_gpu = subprocess.run(gpu_cmd, capture_output=True, text=True)
    t_gpu = time.perf_counter() - t0
    if res_gpu.returncode != 0:
        raise RuntimeError(f"GPU render failed for {cell_id}:\n{res_gpu.stderr}\n{res_gpu.stdout}")

    # rt_view diff
    diff_cmd = [view_bin, cpu_ppm, "--diff", gpu_ppm, "--stats"]
    res_view = subprocess.run(diff_cmd, capture_output=True, text=True)
    if res_view.returncode != 0:
        raise RuntimeError(f"rt_view failed for {cell_id}:\n{res_view.stderr}\n{res_view.stdout}")

    # parse diff output: "diff mean=0.173991 max=137 over8=0.292969%"
    m = re.search(r"diff mean=([\d\.]+) max=(\d+) over8=([\d\.]+)%", res_view.stdout)
    if not m:
        raise ValueError(f"Could not parse rt_view output:\n{res_view.stdout}")

    mean_abs = float(m.group(1))
    max_abs = int(m.group(2))
    over8 = float(m.group(3))

    return {
        "scene": scene,
        "mode": mode,
        "depth": depth,
        "rng": rng_mode,
        "spp": spp,
        "w": w,
        "h": h,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "over8": over8,
        "t_cpu": round(t_cpu, 3),
        "t_gpu": round(t_gpu, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimator Parity Drift Harness")
    parser.add_argument("--fast", action="store_true", help="Run fast subset (default scene, 64x36, 4spp, under 10s budget)")
    parser.add_argument("--scenes", default=None, help="Comma-separated scenes to test")
    parser.add_argument("--modes", default="mis,mixture", help="Sampling modes (default: mis,mixture)")
    parser.add_argument("--depths", default="1,2,50", help="Bounce depths (default: 1,2,50)")
    parser.add_argument("--rng", default="fixed,sampled", help="RNG modes (default: fixed,sampled)")
    parser.add_argument("--spp", type=int, default=None, help="Samples per pixel")
    parser.add_argument("--width", type=int, default=None, help="Resolution width")
    parser.add_argument("--height", type=int, default=None, help="Resolution height")
    parser.add_argument("--cpu-bin", default=None, help="Path to raytracer CPU executable")
    parser.add_argument("--gpu-bin", default=None, help="Path to rt_gpu executable")
    parser.add_argument("--spv-bin", default=None, help="Path to path.spv compute shader")
    parser.add_argument("--view-bin", default=None, help="Path to rt_view executable")
    parser.add_argument("--out-dir", default=".scratch/parity_runs", help="Directory for generated PPMs")
    parser.add_argument("--save-baseline", default=None, help="Path to save baseline JSON")
    parser.add_argument("--compare-baseline", default=None, help="Path to baseline JSON to compare against")
    parser.add_argument("--tolerance-factor", type=float, default=1.5, help="Multiplier against baseline thresholds (default 1.5)")
    args = parser.parse_args()

    search_dirs = [
        "build/Release",
        "build",
        "bin",
        ".",
    ]
    cpu_bin = args.cpu_bin or find_binary(["raytracer"], search_dirs)
    gpu_bin = args.gpu_bin or find_binary(["rt_gpu"], search_dirs)
    view_bin = args.view_bin or find_binary(["rt_view"], search_dirs)
    spv_bin = args.spv_bin or find_binary(["path.spv"], ["build/shaders", "shaders", "build"])

    if not cpu_bin or not Path(cpu_bin).is_file():
        sys.exit(f"Error: raytracer binary not found in {search_dirs}")
    if not gpu_bin or not Path(gpu_bin).is_file():
        sys.exit(f"Error: rt_gpu binary not found in {search_dirs}")
    if not view_bin or not Path(view_bin).is_file():
        sys.exit(f"Error: rt_view binary not found in {search_dirs}")
    if not spv_bin or not Path(spv_bin).is_file():
        sys.exit(f"Error: path.spv shader not found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probe Vulkan device availability
    probe_ppm = str(out_dir / "probe.ppm")
    probe_res = subprocess.run([
        gpu_bin, spv_bin,
        "--scene", "default",
        "--width", "8", "--height", "8",
        "--spp", "1", "--fixed-rng",
        probe_ppm
    ], capture_output=True, text=True)
    if probe_res.returncode != 0:
        err_msg = probe_res.stderr + probe_res.stdout
        if any(w in err_msg for w in ["VK_", "VkResult", "vkCreateInstance", "vkEnumeratePhysicalDevices"]):
            print(f"Notice: No Vulkan compute device available ({probe_res.stderr.strip()}).")
            print("Skipping parity harness on this headless/driverless host.")
            sys.exit(0)
        else:
            sys.exit(f"Error probing GPU binary:\n{probe_res.stderr}\n{probe_res.stdout}")

    if args.fast:
        scenes = ["default"] if not args.scenes else [s.strip() for s in args.scenes.split(",")]
        modes = ["mis", "mixture"] if not args.modes else [m.strip() for m in args.modes.split(",")]
        depths = [1, 2, 50]
        rng_modes = ["fixed", "sampled"]
        spp = args.spp or 4
        w = args.width or 64
        h = args.height or 36
    else:
        scenes = ["default", "showcase"] if not args.scenes else [s.strip() for s in args.scenes.split(",")]
        modes = [m.strip() for m in args.modes.split(",")]
        depths = [int(d.strip()) for d in args.depths.split(",")]
        rng_modes = [r.strip() for r in args.rng.split(",")]
        spp = args.spp or 4
        w = args.width or 64
        h = args.height or 36

    baseline_data = {}
    baseline_source = args.compare_baseline
    if not baseline_source:
        default_baseline = Path("tools/parity_baseline.json")
        if default_baseline.is_file():
            baseline_source = str(default_baseline)

    if baseline_source:
        base_path = Path(baseline_source)
        if base_path.is_file():
            with open(base_path, "r") as f:
                baseline_data = json.load(f).get("cells", {})
            print(f"Comparing against baseline: {base_path} ({len(baseline_data)} cells)")

    results = {}
    failures = []

    print(f"Running parity harness: {len(scenes)} scenes x {len(modes)} modes x {len(depths)} depths x {len(rng_modes)} RNG modes = {len(scenes)*len(modes)*len(depths)*len(rng_modes)} cells")
    print(f"Resolution: {w}x{h} @ {spp} spp | Output: {out_dir}\n")

    header = f"| Scene | Mode | Depth | RNG | Mean Abs | Max Abs | Over 8 | CPU (s) | GPU (s) | Status |"
    sep    = f"|:------|:-----|:------|:----|:---------|:--------|:-------|:--------|:--------|:-------|"
    print(header)
    print(sep)

    for sc in scenes:
        scene_h = h
        if sc in ["cornell", "book2"] and not args.height:
            scene_h = w

        for md in modes:
            for d in depths:
                for rng in rng_modes:
                    key = f"{sc}/{md}/d{d}/{rng}"
                    cell = run_cell(cpu_bin, gpu_bin, spv_bin, view_bin,
                                    sc, md, d, rng, spp, w, scene_h, out_dir)
                    results[key] = cell

                    cell_passed = True
                    reason = ""

                    if key in baseline_data:
                        b = baseline_data[key]
                        tol = args.tolerance_factor
                        if rng == "fixed":
                            max_allowed_mean = max(b["mean_abs"] * tol, b["mean_abs"] + 0.05)
                            max_allowed_over8 = max(b["over8"] * tol, b["over8"] + 0.2)
                        else:
                            max_allowed_mean = max(b["mean_abs"] * tol, b["mean_abs"] + 2.0)
                            max_allowed_over8 = max(b["over8"] * tol, b["over8"] + 2.0)

                        if cell["mean_abs"] > max_allowed_mean:
                            cell_passed = False
                            reason = f"mean {cell['mean_abs']:.3f} > baseline {b['mean_abs']:.3f} * {tol}"
                        elif cell["over8"] > max_allowed_over8:
                            cell_passed = False
                            reason = f"over8 {cell['over8']:.2f}% > baseline {b['over8']:.2f}% * {tol}"
                    else:
                        if rng == "fixed":
                            if cell["mean_abs"] > 1.0 or cell["over8"] > 3.0:
                                cell_passed = False
                                reason = f"fixed-RNG drift exceeded (mean={cell['mean_abs']:.3f}, over8={cell['over8']:.2f}%)"
                        else:
                            if cell["mean_abs"] > 50.0 or cell["over8"] > 60.0:
                                cell_passed = False
                                reason = f"sampled drift exceeded (mean={cell['mean_abs']:.3f}, over8={cell['over8']:.2f}%)"

                    status = "PASS" if cell_passed else "FAIL"
                    if not cell_passed:
                        failures.append((key, reason))

                    print(f"| {sc:5} | {md:7} | {d:5} | {rng:7} | {cell['mean_abs']:8.3f} | {cell['max_abs']:7} | {cell['over8']:5.2f}% | {cell['t_cpu']:7.3f} | {cell['t_gpu']:7.3f} | {status} |")

    if args.save_baseline:
        save_path = Path(args.save_baseline)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"version": 1, "cells": results}, f, indent=2)
        print(f"\nSaved baseline with {len(results)} cells to {save_path}")
        print("Baseline established successfully.")
        sys.exit(0)

    if failures:
        print(f"\nParity Harness FAILED with {len(failures)} failures:")
        for k, r in failures:
            print(f"  - {k}: {r}")
        sys.exit(1)
    else:
        print(f"\nParity Harness PASSED: all {len(results)} cells verified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
