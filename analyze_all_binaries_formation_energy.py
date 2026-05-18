#!/usr/bin/env python3.11
import os
import re
import csv
import subprocess
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset"
OUT = "/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations/all_binary_fe_diagnostics_20260509"
ATOMIC_NUM = {"V": 23, "Nb": 41, "Ta": 73, "Ti": 22, "Zr": 40, "Hf": 72}
PAT_OUT = re.compile(r"^N(\d+)\.OUTCAR$")
PAT_COMP = re.compile(r"([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)")


@dataclass
class BinSystem:
    dataset: str
    elem1: str
    elem2: str
    ref1: float
    ref2: float


def find_final_outcar(case_dir: str):
    try:
        files = os.listdir(case_dir)
    except Exception:
        return None
    best = None
    best_idx = -1
    for f in files:
        m = PAT_OUT.match(f)
        if m:
            idx = int(m.group(1))
            if idx > best_idx:
                best_idx = idx
                best = f
    if best:
        return os.path.join(case_dir, best)
    if "OUTCAR" in files:
        return os.path.join(case_dir, "OUTCAR")
    return None


def sigma0_energy(outcar_path: str):
    cmd = f"grep -n \"energy(sigma->0) =\" '{outcar_path}' | tail -1 | rev | cut -d ' ' -f1 | rev"
    out = subprocess.getoutput(cmd).strip()
    if not out:
        raise ValueError("No energy(sigma->0) line")
    return float(out)


def converged(outcar_path: str):
    cmd = f"grep -n \"reached required accuracy\" '{outcar_path}' | head -1"
    return subprocess.getoutput(cmd).strip() != ""


def parse_comp_counts(comp: str, e1: str, e2: str):
    m = PAT_COMP.fullmatch(comp)
    if not m:
        return None
    a1, n1, a2, n2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    if {a1, a2} != {e1, e2}:
        return None
    return {a1: n1, a2: n2}


def discover_binary_systems():
    systems = []
    for d in sorted(os.listdir(BASE)):
        ds = os.path.join(BASE, d)
        if not (os.path.isdir(ds) and d.startswith("bcc_")):
            continue

        pure_elems = []
        for lvl1 in sorted(os.listdir(ds)):
            p = os.path.join(ds, lvl1, f"{lvl1}128", "case-1")
            if os.path.isdir(p) and lvl1 in ATOMIC_NUM:
                pure_elems.append(lvl1)

        if len(pure_elems) != 2:
            continue

        e1, e2 = sorted(pure_elems, key=lambda x: ATOMIC_NUM[x])

        out1 = find_final_outcar(os.path.join(ds, e1, f"{e1}128", "case-1"))
        out2 = find_final_outcar(os.path.join(ds, e2, f"{e2}128", "case-1"))
        if not out1 or not out2:
            continue

        ref1 = sigma0_energy(out1) / 128.0
        ref2 = sigma0_energy(out2) / 128.0
        systems.append(BinSystem(d, e1, e2, ref1, ref2))

    return systems


def analyze_system(sys: BinSystem):
    ds = os.path.join(BASE, sys.dataset)
    x_vals = []
    fe_vals = []

    # Dataset structure is typically:
    # <dataset>/<binary-tag>/<composition>/case-*
    for level1 in sorted(os.listdir(ds)):
        level1_dir = os.path.join(ds, level1)
        if not os.path.isdir(level1_dir):
            continue
        for comp in sorted(os.listdir(level1_dir)):
            comp_dir = os.path.join(level1_dir, comp)
            if not os.path.isdir(comp_dir):
                continue

            counts = parse_comp_counts(comp, sys.elem1, sys.elem2)
            if counts is None:
                continue

            n1 = counts.get(sys.elem1, 0)
            n2 = counts.get(sys.elem2, 0)
            n = n1 + n2
            if n == 0:
                continue

            x = n2 / n

            for case in os.listdir(comp_dir):
                case_dir = os.path.join(comp_dir, case)
                if not (os.path.isdir(case_dir) and case.startswith("case-")):
                    continue

                outcar = find_final_outcar(case_dir)
                if not outcar:
                    continue
                try:
                    et = sigma0_energy(outcar)
                except Exception:
                    continue

                fe = ((et - n1 * sys.ref1 - n2 * sys.ref2) / n) * 1000.0
                x_vals.append(x)
                fe_vals.append(fe)

    return np.array(x_vals), np.array(fe_vals)


def write_plots(sys: BinSystem, x: np.ndarray, y: np.ndarray):
    os.makedirs(OUT, exist_ok=True)
    tag = f"{sys.elem1}{sys.elem2}"

    plt.figure(figsize=(8, 6))
    plt.hist(y, bins=100, color="steelblue")
    plt.ylabel("Number of configurations")
    plt.xlabel("Formation Energy (meV/atom)")
    plt.title(f"{sys.dataset}: FE histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"FE_hist_{tag}.png"), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=8)
    plt.xlabel(f"{sys.elem2} fraction")
    plt.ylabel("Formation Energy (meV/atom)")
    plt.title(f"{sys.dataset}: FE vs composition")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"FE_vs_x_{tag}.png"), dpi=300)
    plt.close()


def stats(v):
    return float(np.mean(v)), float(np.min(v)), float(np.max(v)), int(v.size)


def main():
    systems = discover_binary_systems()
    os.makedirs(OUT, exist_ok=True)
    summary_path = os.path.join(OUT, "all_binary_fe_endpoint_summary.csv")

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "elem1", "elem2", "ref1_eVatom", "ref2_eVatom", "n_total",
            "low_n", "low_mean_meVatom", "low_min", "low_max",
            "high_n", "high_mean_meVatom", "high_min", "high_max"
        ])

        for sys in systems:
            x, y = analyze_system(sys)
            if y.size == 0:
                print(f"{sys.dataset}: NO_DATA")
                continue

            write_plots(sys, x, y)

            low = y[x <= 0.1]
            high = y[x >= 0.9]
            lmean, lmin, lmax, ln = stats(low)
            hmean, hmin, hmax, hn = stats(high)

            w.writerow([
                sys.dataset, sys.elem1, sys.elem2,
                f"{sys.ref1:.8f}", f"{sys.ref2:.8f}", int(y.size),
                ln, f"{lmean:.6f}", f"{lmin:.6f}", f"{lmax:.6f}",
                hn, f"{hmean:.6f}", f"{hmin:.6f}", f"{hmax:.6f}"
            ])

            print(
                f"{sys.dataset} ({sys.elem1}-{sys.elem2}) n={y.size} "
                f"low_mean={lmean:.2f} high_mean={hmean:.2f} meV/atom"
            )

    print(f"WROTE_SUMMARY {summary_path}")


if __name__ == "__main__":
    main()
