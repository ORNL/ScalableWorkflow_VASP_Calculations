#!/usr/bin/env python3.11
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ATOMIC_NUM = {'V': 23, 'Nb': 41, 'Ta': 73, 'Ti': 22, 'Zr': 40, 'Hf': 72}


def parse_counts(comp_name, e1, e2):
    m1 = re.search(rf'{e1}(\d+)', comp_name)
    m2 = re.search(rf'{e2}(\d+)', comp_name)
    if m1 and m2:
        return int(m1.group(1)), int(m2.group(1))
    m1 = re.search(rf'{e2}(\d+)', comp_name)
    m2 = re.search(rf'{e1}(\d+)', comp_name)
    if m1 and m2:
        return int(m2.group(1)), int(m1.group(1))
    return None


def load_series(root, e1, e2):
    selected = e1 if ATOMIC_NUM[e1] < ATOMIC_NUM[e2] else e2

    x_fe, y_fe = [], []
    x_rmsd, y_rmsd = [], []

    for comp in sorted(os.listdir(root)):
        comp_dir = os.path.join(root, comp)
        if not os.path.isdir(comp_dir):
            continue
        counts = parse_counts(comp, e1, e2)
        if counts is None:
            continue

        n1, n2 = counts
        total = n1 + n2
        if total == 0:
            continue
        x = (n1 / total) if selected == e1 else (n2 / total)

        for case in os.listdir(comp_dir):
            case_dir = os.path.join(comp_dir, case)
            if not (os.path.isdir(case_dir) and case.startswith('case-')):
                continue

            fe_path = os.path.join(case_dir, 'formation_energy.txt')
            if not os.path.exists(fe_path):
                fe_path = os.path.join(case_dir, 'formation_energy-bis.txt')
            if os.path.exists(fe_path):
                try:
                    v = float(open(fe_path).read().strip())
                    x_fe.append(x)
                    y_fe.append(v)
                except Exception:
                    pass

            rmsd_path = os.path.join(case_dir, 'root_mean_squared_displacement.txt')
            if not os.path.exists(rmsd_path):
                rmsd_path = os.path.join(case_dir, 'root_mean_squared_displacement-bis.txt')
            if os.path.exists(rmsd_path):
                try:
                    v = float(open(rmsd_path).read().strip())
                    x_rmsd.append(x)
                    y_rmsd.append(v)
                except Exception:
                    pass

    return selected, np.array(x_fe), np.array(y_fe), np.array(x_rmsd), np.array(y_rmsd)


def save_plots(root, e1, e2):
    selected, x_fe, y_fe, x_rmsd, y_rmsd = load_series(root, e1, e2)

    if y_fe.size:
        plt.figure(figsize=(8, 6))
        plt.hist(y_fe, bins=100, color='blue')
        plt.ylabel('Number of configurations')
        plt.xlabel('Formation Energy (meV/atom)')
        plt.title(f'{e1}{e2} - BCC phase')
        plt.tight_layout()
        plt.savefig(os.path.join(root, f'BCC_Enthalpy_Histogram_{e1}{e2}.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(x_fe, y_fe, s=8)
        plt.xlabel(f'{selected} concentration')
        plt.ylabel('Formation Energy (meV/atom)')
        plt.title(f'{e1}{e2}')
        plt.tight_layout()
        plt.savefig(os.path.join(root, f'BCC_enthalpy_vs_concentration_{e1}{e2}.png'), dpi=400)
        plt.close()

    if y_rmsd.size:
        plt.figure(figsize=(8, 6))
        plt.hist(y_rmsd, bins=100, color='blue')
        plt.ylabel('Number of configurations')
        plt.xlabel('RMSD (angstrom)')
        plt.title(f'{e1}{e2} - BCC phase')
        plt.tight_layout()
        plt.savefig(os.path.join(root, f'BCC_MSD_Histogram_{e1}{e2}.png'), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(x_rmsd, y_rmsd, s=8)
        plt.xlabel(f'{selected} concentration')
        plt.ylabel('RMSD (angstrom)')
        plt.title(f'{e1}{e2}')
        plt.tight_layout()
        plt.savefig(os.path.join(root, f'BCC_MSD_vs_concentration_{e1}{e2}.png'), dpi=400)
        plt.close()

    return y_fe.size, y_rmsd.size


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('Usage: plot_binary_from_text_outputs.py <root> <E1> <E2>')
    root, e1, e2 = sys.argv[1], sys.argv[2], sys.argv[3]
    nfe, nrmsd = save_plots(root, e1, e2)
    print(f'PLOTTED {e1}-{e2}: FE={nfe} RMSD={nrmsd}')
