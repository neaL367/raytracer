#!/usr/bin/env python3
"""Tolerant image comparison for CPU<->GPU parity (fp64 vs fp32 can never be
bit-exact). Usage: compare_ppm.py <ref.ppm> <got.ppm> [max] [mean] [frac2]
Exits 0 on PASS, 1 on FAIL.
Calibrated 2026-09: normal-shade parity over the 317-prim scene gives
max=39 (two silhouette-edge pixels where fp32/fp64 flip a grazing hit),
mean~0.0001, frac>2LSB~0.001%. Defaults leave small headroom above that.
"""
import sys


def load(path):
    with open(path) as f:
        tokens = []
        for line in f:
            line = line.split("#", 1)[0]
            tokens += line.split()
    assert tokens[0] == "P3", f"{path}: not P3 PPM"
    w, h, mx = int(tokens[1]), int(tokens[2]), int(tokens[3])
    vals = list(map(int, tokens[4:]))
    assert len(vals) == w * h * 3, f"{path}: truncated ({len(vals)} vs {w*h*3})"
    assert mx == 255
    return w, h, vals


def main():
    ref_path, got_path = sys.argv[1], sys.argv[2]
    max_tol = float(sys.argv[3]) if len(sys.argv) > 3 else 48.0
    mean_tol = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
    frac_tol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.005
    rw, rh, rv = load(ref_path)
    gw, gh, gv = load(got_path)
    assert (rw, rh) == (gw, gh), "dimension mismatch"
    n = len(rv)
    maxd = 0
    total = 0
    bad = 0
    for a, b in zip(rv, gv):
        d = abs(a - b)
        total += d
        if d > maxd:
            maxd = d
        if d > 2:
            bad += 1
    mean = total / n
    frac = 100.0 * bad / n
    ok = maxd <= max_tol and mean <= mean_tol and frac <= frac_tol
    print(f"max={maxd} mean={mean:.4f} frac>2LSB={frac:.3f}% -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
