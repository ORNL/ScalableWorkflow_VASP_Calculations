#!/usr/bin/env python3.11
import csv
import os
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


@dataclass
class SystemConfig:
    label: str
    elem1: str
    elem2: str
    input_dir: str


def load_true_endpoints(base_dir):
    endpoint_csv = os.path.join(
        base_dir,
        "corrected_fe_strict_endpoints_20260518",
        "strict_endpoint_check_20260518.csv",
    )
    values = {}
    if not os.path.isfile(endpoint_csv):
        return values

    with open(endpoint_csv, "r", encoding="ascii") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("dataset", "")
            if ds == "bcc_NbZr":
                key = "NbZr"
            elif ds == "bcc_TaZr":
                key = "TaZr"
            elif ds == "bcc_VZr":
                key = "VZr"
            else:
                continue

            try:
                values[key] = (
                    float(row["FE_x0_meVatom"]),
                    float(row["FE_x1_meVatom"]),
                )
            except Exception:
                continue
    return values


def get_color_density(xdata, ydata, nbin=28):
    hist2d, xbins_edge, ybins_edge = np.histogram2d(x=xdata, y=ydata, bins=[nbin, nbin])
    if np.max(hist2d) > 0:
        hist2d = hist2d / np.max(hist2d)

    xbin_cen = 0.5 * (xbins_edge[:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[:-1] + ybins_edge[1:])
    grid_y, grid_x = np.meshgrid(ybin_cen, xbin_cen)

    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = np.reshape(grid_x, len(xbin_cen) * nbin)
    loc_pts[:, 1] = np.reshape(grid_y, len(xbin_cen) * nbin)

    density = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )
    return density


def collect_data(input_dir, elem1, elem2):
    xdata = []
    fe_data = []

    comp_pat_12 = re.compile(rf"({elem1})(\d+)({elem2})(\d+)")
    comp_pat_21 = re.compile(rf"({elem2})(\d+)({elem1})(\d+)")

    for comp_name in sorted(os.listdir(input_dir)):
        comp_dir = os.path.join(input_dir, comp_name)
        if not os.path.isdir(comp_dir):
            continue

        m12 = comp_pat_12.fullmatch(comp_name)
        m21 = comp_pat_21.fullmatch(comp_name)
        if m12 is None and m21 is None:
            continue

        if m12 is not None:
            n1 = int(m12.group(2))
            n2 = int(m12.group(4))
        else:
            n2 = int(m21.group(2))
            n1 = int(m21.group(4))

        n_total = n1 + n2
        if n_total <= 0:
            continue

        x = n2 / n_total

        for case_name in sorted(os.listdir(comp_dir)):
            case_dir = os.path.join(comp_dir, case_name)
            if not (os.path.isdir(case_dir) and case_name.startswith("case-")):
                continue

            fe_file = os.path.join(case_dir, "formation_energy.txt")
            if not os.path.isfile(fe_file):
                continue

            try:
                with open(fe_file, "r", encoding="ascii") as f:
                    fe = float(f.read().strip())
                xdata.append(x)
                fe_data.append(fe)
            except Exception:
                continue

    return np.array(xdata), np.array(fe_data)


def apply_linear_offset(x, fe_raw, x_low_mean, x_high_mean, low_mean, high_mean):
    # Fit offset(x) = a + b*x so that near-end mean FE becomes zero in both bins.
    denom = (x_high_mean - x_low_mean)
    if abs(denom) < 1.0e-12:
        b = 0.0
        a = 0.5 * (low_mean + high_mean)
    else:
        b = (high_mean - low_mean) / denom
        a = low_mean - b * x_low_mean
    offset = a + b * x
    fe_corr = fe_raw - offset
    return offset, fe_corr, a, b


def fit_line_from_two_bins(x_low_mean, x_high_mean, y_low_mean, y_high_mean):
    denom = (x_high_mean - x_low_mean)
    if abs(denom) < 1.0e-12:
        b = 0.0
        a = 0.5 * (y_low_mean + y_high_mean)
    else:
        b = (y_high_mean - y_low_mean) / denom
        a = y_low_mean - b * x_low_mean
    return a, b


def apply_endpoint_alignment_shift(x, fe_stage1, true_x0, true_x1, n_fit=3):
    # Project endpoint behavior from near-end COMPOSITION means.
    # This follows the requested criterion: extrapolate from near-pure compositions.
    uniq = np.array(sorted(set(float(v) for v in x)))
    by_x = {}
    for xv in uniq:
        mask = (np.abs(x - xv) < 1.0e-12)
        by_x[xv] = float(np.mean(fe_stage1[mask]))

    if len(uniq) < 2:
        proj_x0 = float(np.mean(fe_stage1))
        proj_x1 = proj_x0
    else:
        n = min(n_fit, len(uniq))
        x_low = uniq[:n]
        y_low = np.array([by_x[v] for v in x_low])
        x_high = uniq[-n:]
        y_high = np.array([by_x[v] for v in x_high])

        # Use linear extrapolation at each edge.
        if len(x_low) >= 2:
            m_low, b_low = np.polyfit(x_low, y_low, 1)
            proj_x0 = b_low
        else:
            proj_x0 = y_low[0]

        if len(x_high) >= 2:
            m_high, b_high = np.polyfit(x_high, y_high, 1)
            proj_x1 = m_high * 1.0 + b_high
        else:
            proj_x1 = y_high[-1]

    # Shift needed so true endpoints align with projected endpoints.
    d0 = true_x0 - proj_x0
    d1 = true_x1 - proj_x1
    shift = (1.0 - x) * d0 + x * d1
    fe_final = fe_stage1 + shift
    return shift, fe_final, proj_x0, proj_x1, d0, d1


def write_scatter_csv(path, x, fe_raw, offset, fe_corr):
    with open(path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["x_elem2", "fe_raw_meVatom", "offset_meVatom", "fe_offset_corrected_meVatom"])
        for i in range(len(x)):
            w.writerow([f"{x[i]:.8f}", f"{fe_raw[i]:.8f}", f"{offset[i]:.8f}", f"{fe_corr[i]:.8f}"])


def write_scatter_csv_final(path, x, fe_raw, offset1, fe_stage1, shift2, fe_final):
    with open(path, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow([
            "x_elem2",
            "fe_raw_meVatom",
            "offset_stage1_meVatom",
            "fe_stage1_meVatom",
            "shift_stage2_meVatom",
            "fe_final_meVatom",
        ])
        for i in range(len(x)):
            w.writerow([
                f"{x[i]:.8f}",
                f"{fe_raw[i]:.8f}",
                f"{offset1[i]:.8f}",
                f"{fe_stage1[i]:.8f}",
                f"{shift2[i]:.8f}",
                f"{fe_final[i]:.8f}",
            ])


def make_plot(path, title, x, y, xlabel, ylabel):
    plt.figure(figsize=(11, 7))
    density = get_color_density(x, y)
    sc = plt.scatter(x, y, s=10, c=density, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(sc, label="Density")
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=320)
    plt.close()


def main():
    base = "/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations"
    out_dir = os.path.join(base, "offset_corrected_plots_20260518")
    os.makedirs(out_dir, exist_ok=True)

    systems = [
        SystemConfig("NbZr", "Nb", "Zr", os.path.join(base, "bcc_enthalpy_NbZr_meVatom_corrected_20260518")),
        SystemConfig("TaZr", "Ta", "Zr", os.path.join(base, "bcc_enthalpy_TaZr_meVatom_corrected_20260518")),
        SystemConfig("VZr", "V", "Zr", os.path.join(base, "bcc_enthalpy_VZr_meVatom_corrected_20260518")),
    ]

    true_endpoints = load_true_endpoints(base)

    summary_rows = []

    for sys in systems:
        x, fe_raw = collect_data(sys.input_dir, sys.elem1, sys.elem2)
        if x.size == 0:
            print(f"WARNING: No FE data found for {sys.label}")
            continue

        low = fe_raw[x <= 0.1]
        high = fe_raw[x >= 0.9]
        if low.size == 0 or high.size == 0:
            print(f"WARNING: Missing near-end bins for {sys.label}")
            continue

        low_mean = float(np.mean(low))
        high_mean = float(np.mean(high))
        x_low_mean = float(np.mean(x[x <= 0.1]))
        x_high_mean = float(np.mean(x[x >= 0.9]))

        offset, fe_corr, a_off, b_off = apply_linear_offset(
            x,
            fe_raw,
            x_low_mean,
            x_high_mean,
            low_mean,
            high_mean,
        )

        low_corr = fe_corr[x <= 0.1]
        high_corr = fe_corr[x >= 0.9]

        true_x0, true_x1 = true_endpoints.get(sys.label, (0.0, 0.0))
        shift2, fe_final, proj_x0, proj_x1, d0, d1 = apply_endpoint_alignment_shift(
            x,
            fe_corr,
            true_x0,
            true_x1,
        )

        low_final = fe_final[x <= 0.1]
        high_final = fe_final[x >= 0.9]

        write_scatter_csv(
            os.path.join(out_dir, f"offset_corrected_scatter_{sys.label}.csv"),
            x,
            fe_raw,
            offset,
            fe_corr,
        )

        write_scatter_csv_final(
            os.path.join(out_dir, f"offset_twostage_scatter_{sys.label}.csv"),
            x,
            fe_raw,
            offset,
            fe_corr,
            shift2,
            fe_final,
        )

        make_plot(
            os.path.join(out_dir, f"FE_vs_x_{sys.label}_raw_density.png"),
            f"{sys.label} Raw FE (Density)",
            x,
            fe_raw,
            f"{sys.elem2} concentration",
            "Formation Energy (meV/atom)",
        )

        make_plot(
            os.path.join(out_dir, f"FE_vs_x_{sys.label}_offset_corrected_density.png"),
            f"{sys.label} Offset-Corrected FE (Density)",
            x,
            fe_corr,
            f"{sys.elem2} concentration",
            "Offset-Corrected Formation Energy (meV/atom)",
        )

        make_plot(
            os.path.join(out_dir, f"FE_vs_x_{sys.label}_twostage_corrected_density.png"),
            f"{sys.label} Two-Stage Corrected FE (Density)",
            x,
            fe_final,
            f"{sys.elem2} concentration",
            "Two-Stage Corrected Formation Energy (meV/atom)",
        )

        plt.figure(figsize=(9, 6))
        xline = np.array([0.0, 1.0])
        yline = a_off + b_off * xline
        plt.plot(xline, yline, "-", color="darkorange", linewidth=2)
        plt.scatter([x_low_mean, x_high_mean], [low_mean, high_mean], c="darkorange", s=70)
        plt.xlabel(f"{sys.elem2} concentration")
        plt.ylabel("Applied Offset (meV/atom)")
        plt.title(f"{sys.label} Linear Offset Model")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"offset_model_{sys.label}.png"), dpi=300)
        plt.close()

        summary_rows.append([
            sys.label,
            f"{low_mean:.8f}",
            f"{high_mean:.8f}",
            f"{np.mean(low):.8f}",
            f"{np.mean(high):.8f}",
            f"{np.mean(low_corr):.8f}",
            f"{np.mean(high_corr):.8f}",
            f"{np.mean(low_final):.8f}",
            f"{np.mean(high_final):.8f}",
            f"{x_low_mean:.8f}",
            f"{x_high_mean:.8f}",
            f"{a_off:.8f}",
            f"{b_off:.8f}",
            f"{true_x0:.8f}",
            f"{true_x1:.8f}",
            f"{proj_x0:.8f}",
            f"{proj_x1:.8f}",
            f"{d0:.8f}",
            f"{d1:.8f}",
            str(int(x.size)),
        ])

        print(
            f"{sys.label}: near_low raw={np.mean(low):.3f}, near_high raw={np.mean(high):.3f}, "
            f"near_low stage1={np.mean(low_corr):.3f}, near_high stage1={np.mean(high_corr):.3f}, "
            f"near_low final={np.mean(low_final):.3f}, near_high final={np.mean(high_final):.3f}"
        )

    summary_csv = os.path.join(out_dir, "offset_correction_summary.csv")
    with open(summary_csv, "w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow([
            "system",
            "offset_low_mean_meVatom",
            "offset_high_mean_meVatom",
            "near_low_raw_mean_meVatom",
            "near_high_raw_mean_meVatom",
            "near_low_corrected_mean_meVatom",
            "near_high_corrected_mean_meVatom",
            "near_low_twostage_mean_meVatom",
            "near_high_twostage_mean_meVatom",
            "x_low_bin_mean",
            "x_high_bin_mean",
            "offset_intercept_a",
            "offset_slope_b",
            "true_endpoint_x0_meVatom",
            "true_endpoint_x1_meVatom",
            "projected_endpoint_x0_meVatom",
            "projected_endpoint_x1_meVatom",
            "endpoint_shift_d0_meVatom",
            "endpoint_shift_d1_meVatom",
            "n_points",
        ])
        w.writerows(summary_rows)

    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
