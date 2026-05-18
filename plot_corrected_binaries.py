#!/usr/bin/env python3
"""
Plot FE vs composition for corrected (reference-adjusted) binary datasets.
"""
import os
import sys
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.rcParams.update({"font.size": 20})

WORKDIR = "/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations"


def find_outcar(case_dir):
    if not os.path.isdir(case_dir):
        return None
    best = None
    best_idx = -1
    for name in os.listdir(case_dir):
        m = re.match(r"^N(\d+)\.OUTCAR$", name)
        if m:
            idx = int(m.group(1))
            if idx > best_idx:
                best_idx = idx
                best = name
    if best is not None:
        return os.path.join(case_dir, best)
    for candidate in ("OUTCAR", "N0.OUTCAR"):
        p = os.path.join(case_dir, candidate)
        if os.path.isfile(p):
            return p
    return None


def sigma0_total_energy(outcar_path):
    s = subprocess.getoutput(
        f"grep -n 'energy(sigma->0) =' '{outcar_path}' | tail -1 | rev | cut -d ' ' -f1 | rev"
    ).strip()
    if not s:
        raise RuntimeError(f"No energy(sigma->0) found in {outcar_path}")
    return float(s)


def load_reference_overrides(path):
    refs = {}
    with open(path, "r", encoding="ascii") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            el, val = ln.split(",")
            refs[el] = float(val)
    return refs


def infer_true_endpoint_info(elem1, elem2):
    # x=0 corresponds to pure elem1, x=1 to pure elem2.
    if (elem1, elem2) == ("Nb", "Zr"):
        return {
            "ref_csv": os.path.join(WORKDIR, "ref_overrides_bcc_NbZr_NEW.csv"),
            "pure_x0_dir": os.path.join(
                "/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset",
                "bcc_NbZr",
                "Nb",
                "Nb128",
                "case-1",
            ),
            "pure_x1_dir": os.path.join(WORKDIR, "corrected_DFT_pure_element_runs", "bcc_NbZr_Zr128"),
        }
    if (elem1, elem2) == ("Ta", "Zr"):
        return {
            "ref_csv": os.path.join(WORKDIR, "ref_overrides_bcc_TaZr_NEW.csv"),
            "pure_x0_dir": os.path.join(
                "/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset",
                "bcc_TaZr",
                "Ta",
                "Ta128",
                "case-1",
            ),
            "pure_x1_dir": os.path.join(WORKDIR, "corrected_DFT_pure_element_runs", "bcc_TaZr_Zr128"),
        }
    if (elem1, elem2) == ("V", "Zr"):
        return {
            "ref_csv": os.path.join(WORKDIR, "ref_overrides_bcc_VZr_NEW.csv"),
            "pure_x0_dir": os.path.join(WORKDIR, "corrected_DFT_pure_element_runs", "bcc_VZr_V128"),
            "pure_x1_dir": os.path.join(WORKDIR, "corrected_DFT_pure_element_runs", "bcc_VZr_Zr128"),
        }
    return None


def compute_true_endpoints(elem1, elem2):
    info = infer_true_endpoint_info(elem1, elem2)
    if info is None:
        return None

    refs = load_reference_overrides(info["ref_csv"])
    out_x0 = find_outcar(info["pure_x0_dir"])
    out_x1 = find_outcar(info["pure_x1_dir"])
    if out_x0 is None or out_x1 is None:
        return None

    et_x0 = sigma0_total_energy(out_x0)
    et_x1 = sigma0_total_energy(out_x1)
    fe_x0 = ((et_x0 - 128 * refs[elem1]) / 128.0) * 1000.0
    fe_x1 = ((et_x1 - 128 * refs[elem2]) / 128.0) * 1000.0

    return {
        "fe_x0": fe_x0,
        "fe_x1": fe_x1,
        "pure_x0_outcar": out_x0,
        "pure_x1_outcar": out_x1,
        "ref_csv": info["ref_csv"],
    }

def getcolordensity(xdata, ydata):
    """Compute 2D histogram density for color mapping"""
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
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


def plot_corrected_binary(input_dir, output_dir, elem1, elem2):
    """Generate plots for corrected binary dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data from all case directories
    xdata = []
    fe_data = []
    
    # Walk through nested structure: input_dir/X-Y/<comp>/case-*/formation_energy.txt
    # or flat structure: input_dir/<comp>/case-*/formation_energy.txt
    xy_dir = os.path.join(input_dir, f"{elem1}-{elem2}")
    if not os.path.isdir(xy_dir):
        xy_dir = os.path.join(input_dir, f"{elem2}-{elem1}")
    
    if not os.path.isdir(xy_dir):
        # Fallback to flat layout used by recomputed output directories.
        xy_dir = input_dir
    
    for comp_dir_name in sorted(os.listdir(xy_dir)):
        comp_dir = os.path.join(xy_dir, comp_dir_name)
        if not os.path.isdir(comp_dir):
            continue
        
        # Parse composition to get x value (fraction of elem2)
        # E.g., "Nb32Zr96" -> Nb=32, Zr=96 -> x_Zr = 96/128
        match = re.match(rf'({elem1})(\d+)({elem2})(\d+)', comp_dir_name)
        if not match:
            match = re.match(rf'({elem2})(\d+)({elem1})(\d+)', comp_dir_name)
            if not match:
                continue
            n2, n1 = int(match.group(2)), int(match.group(4))
        else:
            n1, n2 = int(match.group(2)), int(match.group(4))
        
        n_total = n1 + n2
        x = n2 / n_total  # Fraction of elem2
        
        # Collect FE from all cases in this composition
        for case_dir_name in sorted(os.listdir(comp_dir)):
            case_dir = os.path.join(comp_dir, case_dir_name)
            if not os.path.isdir(case_dir) or not case_dir_name.startswith("case-"):
                continue
            
            fe_file = os.path.join(case_dir, "formation_energy.txt")
            if os.path.isfile(fe_file):
                try:
                    with open(fe_file, 'r') as f:
                        fe = float(f.read().strip())
                    xdata.append(x)
                    fe_data.append(fe)
                except:
                    pass
    
    if len(xdata) == 0:
        print(f"ERROR: No formation energy data found in {input_dir}")
        return False
    
    print(f"{elem1}-{elem2}: n={len(xdata)}, FE range=[{min(fe_data):.2f}, {max(fe_data):.2f}] meV/atom")
    
    # Plot histogram
    plt.figure(figsize=(11, 7))
    plt.hist(fe_data, color="blue", density=False, bins=100)
    plt.ylabel('Number of configurations')
    plt.xlabel('Formation Energy (meV/atom)')
    plt.title(f'{elem1}{elem2} - BCC phase (Corrected References)')
    plt.tight_layout()
    hist_file = os.path.join(output_dir, f'BCC_Enthalpy_Histogram_{elem1}{elem2}_corrected.png')
    plt.savefig(hist_file, dpi=300)
    plt.close()
    print(f"Saved histogram: {hist_file}")
    
    # Plot vs composition
    fig, ax = plt.subplots(figsize=(11, 7))
    hist2d_norm = getcolordensity(xdata, fe_data)
    
    scatter = plt.scatter(xdata, fe_data, s=10, c=hist2d_norm, vmin=0, vmax=1, cmap='viridis')
    cbar = plt.colorbar(scatter, label='Density')
    cbar.ax.tick_params(labelsize=12)

    endpoint_info = compute_true_endpoints(elem1, elem2)
    if endpoint_info is not None:
        plt.scatter(
            [0.0, 1.0],
            [endpoint_info["fe_x0"], endpoint_info["fe_x1"]],
            s=180,
            marker='*',
            c='crimson',
            edgecolors='black',
            linewidths=0.8,
            label='True pure endpoints',
            zorder=5,
        )

    plt.axhline(0.0, color='black', linewidth=1.0, alpha=0.6)
    plt.xlabel(f"{elem2} concentration")
    plt.ylabel('Formation Energy (meV/atom)')
    plt.title(f'{elem1}{elem2} - BCC phase (Corrected References)')
    ax.set_xticks([0.0, 0.5, 1.0])
    if endpoint_info is not None:
        plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'BCC_enthalpy_vs_concentration_{elem1}{elem2}_corrected.png')
    plt.savefig(plot_file, dpi=320)
    plt.close()
    print(f"Saved composition plot: {plot_file}")

    if endpoint_info is not None:
        return {
            "dataset": f"{elem1}{elem2}",
            "elem1": elem1,
            "elem2": elem2,
            "fe_x0_meVatom": endpoint_info["fe_x0"],
            "fe_x1_meVatom": endpoint_info["fe_x1"],
            "pure_x0_outcar": endpoint_info["pure_x0_outcar"],
            "pure_x1_outcar": endpoint_info["pure_x1_outcar"],
            "ref_csv": endpoint_info["ref_csv"],
        }
    return None


if __name__ == "__main__":
    # Optional usage:
    #   python plot_corrected_binaries.py <output_dir> <nbzr_dir> <tazr_dir> <vzr_dir>
    if len(sys.argv) == 5:
        output_base = sys.argv[1]
        binaries = [
            (sys.argv[2], "Nb", "Zr"),
            (sys.argv[3], "Ta", "Zr"),
            (sys.argv[4], "V", "Zr"),
        ]
    else:
        binaries = [
            ("bcc_enthalpy_NbZr_meVatom_corrected_20260510", "Nb", "Zr"),
            ("bcc_enthalpy_TaZr_meVatom_corrected_20260510", "Ta", "Zr"),
            ("bcc_enthalpy_VZr_meVatom_corrected_20260510", "V", "Zr"),
        ]
        output_base = "corrected_plots_20260510"

    os.makedirs(output_base, exist_ok=True)
    
    endpoint_rows = []

    for data_dir, e1, e2 in binaries:
        if os.path.isdir(data_dir):
            row = plot_corrected_binary(data_dir, output_base, e1, e2)
            if row is not None:
                endpoint_rows.append(row)
        else:
            print(f"WARNING: Directory not found: {data_dir}")

    if endpoint_rows:
        endpoint_csv = os.path.join(output_base, "true_endpoints_used_in_plots.csv")
        with open(endpoint_csv, "w", encoding="ascii") as f:
            f.write("dataset,elem1,elem2,fe_x0_meVatom,fe_x1_meVatom,pure_x0_outcar,pure_x1_outcar,ref_csv\n")
            for r in endpoint_rows:
                f.write(
                    f"{r['dataset']},{r['elem1']},{r['elem2']},"
                    f"{r['fe_x0_meVatom']:.8f},{r['fe_x1_meVatom']:.8f},"
                    f"{r['pure_x0_outcar']},{r['pure_x1_outcar']},{r['ref_csv']}\n"
                )
        print(f"Saved endpoint metadata: {endpoint_csv}")
