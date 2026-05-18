#!/usr/bin/env python3.11
import csv
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

BASE = "/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset"
OUT = "/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations/nonzr_fe_outlier_analysis_20260511"
ATOMIC_NUM = {"V": 23, "Nb": 41, "Ta": 73, "Ti": 22, "Zr": 40, "Hf": 72}
PAT_OUT = re.compile(r"^N(\d+)\.OUTCAR$")
PAT_COMP = re.compile(r"([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)")

MAD_Z_THRESHOLD = 3.5
MIN_GROUP_SIZE_FOR_MAD = 8
IQR_MULTIPLIER = 3.0

# Published systems to skip from new filtering/plot generation.
EXCLUDED_DATASETS = {"bcc_NbTa", "bcc_NbV", "bcc_TaV"}


@dataclass
class BinSystem:
    dataset: str
    elem1: str
    elem2: str
    ref1: float
    ref2: float


@dataclass
class CaseRecord:
    dataset: str
    composition: str
    case: str
    case_dir: str
    outcar: str
    x_elem2: float
    fe_mev_atom: float


def find_final_outcar(case_dir: str):
    try:
        files = os.listdir(case_dir)
    except Exception:
        return None
    best = None
    best_idx = -1
    for name in files:
        match = PAT_OUT.match(name)
        if match:
            idx = int(match.group(1))
            if idx > best_idx:
                best_idx = idx
                best = name
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


def parse_comp_counts(comp: str, e1: str, e2: str):
    match = PAT_COMP.fullmatch(comp)
    if not match:
        return None
    a1, n1, a2, n2 = match.group(1), int(match.group(2)), match.group(3), int(match.group(4))
    if {a1, a2} != {e1, e2}:
        return None
    return {a1: n1, a2: n2}


def discover_nonzr_binary_systems():
    systems = []
    for dataset in sorted(os.listdir(BASE)):
        ds_path = os.path.join(BASE, dataset)
        if not (os.path.isdir(ds_path) and dataset.startswith("bcc_")):
            continue
        if dataset in EXCLUDED_DATASETS:
            continue

        pure_elems = []
        for lvl1 in sorted(os.listdir(ds_path)):
            pure_case = os.path.join(ds_path, lvl1, f"{lvl1}128", "case-1")
            if os.path.isdir(pure_case) and lvl1 in ATOMIC_NUM:
                pure_elems.append(lvl1)

        if len(pure_elems) != 2:
            continue
        if "Zr" in pure_elems:
            continue

        elem1, elem2 = sorted(pure_elems, key=lambda x: ATOMIC_NUM[x])
        out1 = find_final_outcar(os.path.join(ds_path, elem1, f"{elem1}128", "case-1"))
        out2 = find_final_outcar(os.path.join(ds_path, elem2, f"{elem2}128", "case-1"))
        if not out1 or not out2:
            continue

        ref1 = sigma0_energy(out1) / 128.0
        ref2 = sigma0_energy(out2) / 128.0
        systems.append(BinSystem(dataset, elem1, elem2, ref1, ref2))
    return systems


def getcolordensity(xdata, ydata):
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    max_hist = np.amax(hist2d)
    if max_hist <= 0:
        return np.zeros_like(xdata)

    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    bcty, bctx = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / max_hist

    bctx1d = np.reshape(bctx, len(xbin_cen) * nbin)
    bcty1d = np.reshape(bcty, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d

    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )
    return hist2d_norm


def collect_case_records(system: BinSystem):
    dataset_dir = os.path.join(BASE, system.dataset)
    records = []

    for level1 in sorted(os.listdir(dataset_dir)):
        level1_dir = os.path.join(dataset_dir, level1)
        if not os.path.isdir(level1_dir):
            continue

        for comp in sorted(os.listdir(level1_dir)):
            comp_dir = os.path.join(level1_dir, comp)
            if not os.path.isdir(comp_dir):
                continue

            counts = parse_comp_counts(comp, system.elem1, system.elem2)
            if counts is None:
                continue

            n1 = counts.get(system.elem1, 0)
            n2 = counts.get(system.elem2, 0)
            total_atoms = n1 + n2
            if total_atoms == 0:
                continue

            x_elem2 = n2 / total_atoms

            for case in sorted(os.listdir(comp_dir)):
                case_dir = os.path.join(comp_dir, case)
                if not (os.path.isdir(case_dir) and case.startswith("case-")):
                    continue

                outcar = find_final_outcar(case_dir)
                if not outcar:
                    continue
                try:
                    total_energy = sigma0_energy(outcar)
                except Exception:
                    continue

                fe_mev_atom = ((total_energy - n1 * system.ref1 - n2 * system.ref2) / total_atoms) * 1000.0
                records.append(
                    CaseRecord(
                        dataset=system.dataset,
                        composition=comp,
                        case=case,
                        case_dir=case_dir,
                        outcar=outcar,
                        x_elem2=x_elem2,
                        fe_mev_atom=fe_mev_atom,
                    )
                )
    return records


def robust_outlier_mask(values: np.ndarray):
    if values.size < MIN_GROUP_SIZE_FOR_MAD:
        return np.zeros(values.size, dtype=bool), "too_few_cases"

    median = float(np.median(values))
    abs_dev = np.abs(values - median)
    mad = float(np.median(abs_dev))
    if mad > 0.0:
        robust_z = 0.6745 * (values - median) / mad
        return np.abs(robust_z) > MAD_Z_THRESHOLD, "mad"

    q1 = float(np.percentile(values, 25.0))
    q3 = float(np.percentile(values, 75.0))
    iqr = q3 - q1
    if iqr > 0.0:
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        return (values < lower) | (values > upper), "iqr"

    return np.zeros(values.size, dtype=bool), "zero_spread"


def write_plots(system: BinSystem, kept_records, flagged_records):
    os.makedirs(OUT, exist_ok=True)
    tag = f"{system.elem1}{system.elem2}"

    x_kept = np.array([r.x_elem2 for r in kept_records]) if kept_records else np.array([])
    y_kept = np.array([r.fe_mev_atom for r in kept_records]) if kept_records else np.array([])
    x_flagged = np.array([r.x_elem2 for r in flagged_records]) if flagged_records else np.array([])
    y_flagged = np.array([r.fe_mev_atom for r in flagged_records]) if flagged_records else np.array([])

    plt.figure(figsize=(8, 6))
    if kept_records:
        density = getcolordensity(x_kept, y_kept)
        sc = plt.scatter(x_kept, y_kept, s=8, c=density, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(sc, label="Density")
    plt.xlabel(f"{system.elem2} fraction")
    plt.ylabel("Formation Energy (meV/atom)")
    plt.title(f"{system.dataset}: FE vs composition (outliers removed)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"FE_vs_x_filtered_{tag}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    if kept_records:
        plt.hist(y_kept, bins=80, color="steelblue", alpha=0.85, label="kept")
    plt.xlabel("Formation Energy (meV/atom)")
    plt.ylabel("Number of configurations")
    plt.title(f"{system.dataset}: FE histogram (outliers removed)")
    if kept_records:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"FE_hist_filtered_{tag}.png"), dpi=250)
    plt.close()


def analyze_system(system: BinSystem):
    records = collect_case_records(system)
    by_comp = defaultdict(list)
    for record in records:
        by_comp[record.composition].append(record)

    flagged_records = []
    kept_records = []
    composition_rows = []

    for composition in sorted(by_comp):
        comp_records = by_comp[composition]
        values = np.array([r.fe_mev_atom for r in comp_records], dtype=float)
        mask, method = robust_outlier_mask(values)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        q1 = float(np.percentile(values, 25.0))
        q3 = float(np.percentile(values, 75.0))
        iqr = q3 - q1

        flagged_count = int(np.sum(mask))
        composition_rows.append([
            system.dataset,
            composition,
            len(comp_records),
            flagged_count,
            method,
            f"{np.min(values):.6f}",
            f"{median:.6f}",
            f"{np.max(values):.6f}",
            f"{mad:.6f}",
            f"{iqr:.6f}",
        ])

        for record, is_flagged in zip(comp_records, mask):
            if is_flagged:
                flagged_records.append((record, median, mad, q1, q3, method))
            else:
                kept_records.append(record)

    return kept_records, flagged_records, composition_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    systems = discover_nonzr_binary_systems()

    summary_csv = os.path.join(OUT, "nonzr_fe_outlier_summary.csv")
    flagged_csv = os.path.join(OUT, "nonzr_fe_flagged_cases.csv")
    per_comp_csv = os.path.join(OUT, "nonzr_fe_per_composition_stats.csv")

    with open(summary_csv, "w", newline="") as f_summary, open(flagged_csv, "w", newline="") as f_flagged, open(per_comp_csv, "w", newline="") as f_comp:
        w_summary = csv.writer(f_summary)
        w_flagged = csv.writer(f_flagged)
        w_comp = csv.writer(f_comp)

        w_summary.writerow([
            "dataset", "elem1", "elem2", "n_total", "n_kept", "n_flagged", "flagged_fraction"
        ])
        w_flagged.writerow([
            "dataset", "composition", "case", "case_dir", "outcar", "x_elem2", "fe_meVatom",
            "composition_median_meVatom", "composition_mad_meVatom", "composition_q1_meVatom",
            "composition_q3_meVatom", "flag_method"
        ])
        w_comp.writerow([
            "dataset", "composition", "n_cases", "n_flagged", "flag_method",
            "min_fe_meVatom", "median_fe_meVatom", "max_fe_meVatom", "mad_fe_meVatom", "iqr_fe_meVatom"
        ])

        for system in systems:
            kept_records, flagged_records, composition_rows = analyze_system(system)
            n_total = len(kept_records) + len(flagged_records)
            n_flagged = len(flagged_records)
            n_kept = len(kept_records)
            frac = (n_flagged / n_total) if n_total else 0.0

            w_summary.writerow([
                system.dataset, system.elem1, system.elem2, n_total, n_kept, n_flagged, f"{frac:.6f}"
            ])

            for row in composition_rows:
                w_comp.writerow(row)

            flagged_case_records = []
            for record, median, mad, q1, q3, method in flagged_records:
                w_flagged.writerow([
                    record.dataset, record.composition, record.case, record.case_dir, record.outcar,
                    f"{record.x_elem2:.6f}", f"{record.fe_mev_atom:.6f}",
                    f"{median:.6f}", f"{mad:.6f}", f"{q1:.6f}", f"{q3:.6f}", method
                ])
                flagged_case_records.append(record)

            write_plots(system, kept_records, flagged_case_records)
            print(
                f"{system.dataset} ({system.elem1}-{system.elem2}) total={n_total} "
                f"flagged={n_flagged} kept={n_kept} frac={frac:.3%}"
            )

    print(f"WROTE {summary_csv}")
    print(f"WROTE {flagged_csv}")
    print(f"WROTE {per_comp_csv}")


if __name__ == "__main__":
    main()
