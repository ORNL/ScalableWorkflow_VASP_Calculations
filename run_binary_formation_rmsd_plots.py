#!/usr/bin/env python3.11
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ATOMIC_NUMBERS = {
    "V": 23,
    "Nb": 41,
    "Ta": 73,
    "Ti": 22,
    "Zr": 40,
    "Hf": 72,
}

DATASET_ROOT = "/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset"
WORK_ROOT = "/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations"
OUT_ROOT = os.path.join(WORK_ROOT, "plots_binary_formation_rmsd_20260509")

ALLOYS = [
    ("Nb", "Zr", "bcc_NbZr"),
    ("Ta", "Zr", "bcc_TaZr"),
    ("V", "Zr", "bcc_VZr"),
]


@dataclass
class CaseResult:
    case_path: str
    concentration_selected: float
    formation_energy_ev_per_atom: float
    rmsd_angstrom: float


def find_final_numbered_file(directory: str, suffix: str) -> Optional[str]:
    pattern = re.compile(rf"^N(\d+)\.{re.escape(suffix)}$")
    max_idx = -1
    max_file = None

    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        return None

    for name in files:
        match = pattern.match(name)
        if not match:
            continue
        idx = int(match.group(1))
        if idx > max_idx:
            max_idx = idx
            max_file = name

    if max_file is not None:
        return max_file

    plain = suffix
    return plain if os.path.exists(os.path.join(directory, plain)) else None


def read_last_sigma0_energy(outcar_path: str) -> Optional[float]:
    # Follow utility behavior: use the final "energy(sigma->0) =" value from OUTCAR.
    target = "energy(sigma->0) ="
    try:
        with open(outcar_path, "r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None

    for line in reversed(lines):
        if target in line:
            parts = line.strip().split()
            try:
                return float(parts[-1])
            except (ValueError, IndexError):
                continue
    return None


def parse_poscar(path: str) -> Tuple[np.ndarray, List[str], List[int], np.ndarray]:
    with open(path, "r") as f:
        raw = [line.rstrip("\n") for line in f]

    scale = float(raw[1].strip())
    lattice = np.array([[float(x) for x in raw[i].split()] for i in range(2, 5)], dtype=float) * scale

    row5 = raw[5].split()
    has_symbols = not all(token.lstrip("+-").isdigit() for token in row5)

    if has_symbols:
        symbols = row5
        counts = [int(x) for x in raw[6].split()]
        coord_start = 7
    else:
        counts = [int(x) for x in row5]
        symbols = [f"E{i}" for i in range(len(counts))]
        coord_start = 6

    if raw[coord_start].strip().lower().startswith("selective"):
        coord_start += 1

    # Coordinates can be Direct or Cartesian. Dataset here uses Direct.
    coordinate_mode = raw[coord_start].strip().lower()
    coord_start += 1

    natoms = sum(counts)
    coords = []
    for i in range(coord_start, coord_start + natoms):
        coords.append([float(x) for x in raw[i].split()[:3]])
    coords = np.array(coords, dtype=float)

    if coordinate_mode.startswith("cart"):
        # Convert Cartesian to fractional using lattice inverse.
        coords = coords @ np.linalg.inv(lattice)

    return lattice, symbols, counts, coords


def compute_case_rmsd(initial_poscar: str, final_contcar: str) -> float:
    lattice, _, _, init_frac = parse_poscar(initial_poscar)
    _, _, _, final_frac = parse_poscar(final_contcar)

    if init_frac.shape != final_frac.shape:
        raise ValueError("Mismatched atom counts between initial and final structures")

    delta = final_frac - init_frac
    delta[delta > 0.9] -= 1.0
    delta[delta < -0.9] += 1.0

    cart_disp = (lattice @ delta.T).T
    sq_norm = np.sum(cart_disp ** 2, axis=1)
    return float(np.sqrt(np.mean(sq_norm)))


def gather_pure_element_energies(alloy_root: str, elements: Tuple[str, str]) -> Dict[str, float]:
    pure = {}
    for elem in elements:
        elem_root = os.path.join(alloy_root, elem)
        if not os.path.isdir(elem_root):
            raise FileNotFoundError(f"Missing pure-element directory: {elem_root}")

        subdirs = [d for d in os.listdir(elem_root) if os.path.isdir(os.path.join(elem_root, d))]
        if not subdirs:
            raise FileNotFoundError(f"No pure-element supercell directory under {elem_root}")

        pure_dir = os.path.join(elem_root, sorted(subdirs)[0], "case-1")
        outcar_name = find_final_numbered_file(pure_dir, "OUTCAR")
        if outcar_name is None:
            raise FileNotFoundError(f"No OUTCAR found in {pure_dir}")

        energy_total = read_last_sigma0_energy(os.path.join(pure_dir, outcar_name))
        if energy_total is None:
            raise ValueError(f"Failed to parse energy from {pure_dir}/{outcar_name}")

        poscar_path = os.path.join(pure_dir, "0.POSCAR")
        if not os.path.exists(poscar_path):
            poscar_path = os.path.join(pure_dir, "POSCAR")
        _, _, counts, _ = parse_poscar(poscar_path)
        natoms = sum(counts)

        pure[elem] = energy_total / natoms
    return pure


def composition_fraction(symbols: List[str], counts: List[int], element: str) -> float:
    total = sum(counts)
    count = 0
    for sym, num in zip(symbols, counts):
        if sym == element:
            count += num
    return count / total


def process_alloy(e1: str, e2: str, alloy_dir_name: str, out_dir: str) -> List[CaseResult]:
    alloy_root = os.path.join(DATASET_ROOT, alloy_dir_name)
    mix_dir = os.path.join(alloy_root, f"{e1}-{e2}")
    if not os.path.isdir(mix_dir):
        raise FileNotFoundError(f"Missing mixture directory: {mix_dir}")

    pure_energies = gather_pure_element_energies(alloy_root, (e1, e2))
    selected = e1 if ATOMIC_NUMBERS[e1] < ATOMIC_NUMBERS[e2] else e2

    results: List[CaseResult] = []

    for composition in sorted(os.listdir(mix_dir)):
        composition_dir = os.path.join(mix_dir, composition)
        if not os.path.isdir(composition_dir):
            continue

        for case in sorted(os.listdir(composition_dir)):
            case_dir = os.path.join(composition_dir, case)
            if not (os.path.isdir(case_dir) and case.startswith("case-")):
                continue

            outcar_name = find_final_numbered_file(case_dir, "OUTCAR")
            if outcar_name is None:
                continue
            outcar_path = os.path.join(case_dir, outcar_name)

            total_energy = read_last_sigma0_energy(outcar_path)
            if total_energy is None:
                continue

            initial_poscar = os.path.join(case_dir, "0.POSCAR")
            if not os.path.exists(initial_poscar):
                initial_poscar = os.path.join(case_dir, "POSCAR")
            if not os.path.exists(initial_poscar):
                continue

            contcar_name = find_final_numbered_file(case_dir, "CONTCAR")
            if contcar_name is None:
                continue
            final_contcar = os.path.join(case_dir, contcar_name)

            lattice, symbols, counts, _ = parse_poscar(initial_poscar)
            _ = lattice  # Keep parse for validation and consistency.
            natoms = sum(counts)

            ref_energy = 0.0
            for sym, count in zip(symbols, counts):
                if sym not in pure_energies:
                    # Ignore unexpected tags if present.
                    continue
                ref_energy += count * pure_energies[sym]

            formation_e_per_atom = (total_energy - ref_energy) / natoms
            rmsd = compute_case_rmsd(initial_poscar, final_contcar)
            x = composition_fraction(symbols, counts, selected)

            results.append(
                CaseResult(
                    case_path=case_dir,
                    concentration_selected=x,
                    formation_energy_ev_per_atom=formation_e_per_atom,
                    rmsd_angstrom=rmsd,
                )
            )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{e1}{e2}_formation_rmsd.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_path",
            f"x_{selected}",
            "formation_energy_eV_per_atom",
            "formation_energy_meV_per_atom",
            "rmsd_angstrom",
        ])
        for row in results:
            writer.writerow([
                row.case_path,
                row.concentration_selected,
                row.formation_energy_ev_per_atom,
                row.formation_energy_ev_per_atom * 1000.0,
                row.rmsd_angstrom,
            ])

    return results


def save_plots(e1: str, e2: str, results: List[CaseResult], out_dir: str) -> None:
    if not results:
        raise RuntimeError(f"No valid cases found for {e1}-{e2}")

    selected = e1 if ATOMIC_NUMBERS[e1] < ATOMIC_NUMBERS[e2] else e2

    x = np.array([r.concentration_selected for r in results], dtype=float)
    fe_mev = np.array([r.formation_energy_ev_per_atom * 1000.0 for r in results], dtype=float)
    rmsd = np.array([r.rmsd_angstrom for r in results], dtype=float)

    plt.figure(figsize=(8, 6))
    plt.hist(fe_mev, bins=100, color="steelblue")
    plt.xlabel("Formation Energy (meV/atom)")
    plt.ylabel("Number of configurations")
    plt.title(f"{e1}-{e2} BCC")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"BCC_Enthalpy_Histogram_{e1}{e2}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(x, fe_mev, s=8, alpha=0.8)
    plt.xlabel(f"{selected} concentration")
    plt.ylabel("Formation Energy (meV/atom)")
    plt.title(f"{e1}-{e2}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"BCC_enthalpy_vs_concentration_{e1}{e2}.png"), dpi=400)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(rmsd, bins=100, color="steelblue")
    plt.xlabel("RMSD (angstrom)")
    plt.ylabel("Number of configurations")
    plt.title(f"{e1}-{e2} BCC")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"BCC_MSD_Histogram_{e1}{e2}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(x, rmsd, s=8, alpha=0.8)
    plt.xlabel(f"{selected} concentration")
    plt.ylabel("RMSD (angstrom)")
    plt.title(f"{e1}-{e2}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"BCC_MSD_vs_concentration_{e1}{e2}.png"), dpi=400)
    plt.close()


def main() -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)

    for e1, e2, alloy_dir_name in ALLOYS:
        alloy_out = os.path.join(OUT_ROOT, f"{e1}{e2}")
        print(f"Processing {e1}-{e2} from {alloy_dir_name}")
        results = process_alloy(e1, e2, alloy_dir_name, alloy_out)
        save_plots(e1, e2, results, alloy_out)
        print(f"  cases used: {len(results)}")
        print(f"  outputs: {alloy_out}")


if __name__ == "__main__":
    main()
