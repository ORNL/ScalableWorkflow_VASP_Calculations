#!/usr/bin/env python3.11
import sys
from compute_root_mean_squared_displacement import compute_mean_squared_displacement

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: run_compute_rmsd.py <source_path> <destination_path>")
    compute_mean_squared_displacement(sys.argv[1], sys.argv[2])
