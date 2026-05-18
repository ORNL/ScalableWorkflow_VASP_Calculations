#!/usr/bin/env python3.11
import os
import sys
import matplotlib
matplotlib.use("Agg")

from plot_enthalpy_binary_raw_data import plot_data as plot_enthalpy
from plot_mean_squared_displacement_binary import plot_data as plot_msd

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: run_plots_binary.py <source_path> <element1> <element2>")

    source_path = sys.argv[1]
    e1 = sys.argv[2]
    e2 = sys.argv[3]

    os.chdir(source_path)
    plot_enthalpy(source_path, [e1, e2])
    plot_msd(source_path, [e1, e2])
