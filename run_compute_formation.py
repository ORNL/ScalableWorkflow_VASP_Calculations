#!/usr/bin/env python3.11
import sys
from compute_enthalpy_ase_object import compute_formation_enthalpy, load_reference_overrides_csv

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: run_compute_formation.py <source_path> <destination_path> [reference_overrides_csv]"
        )

    source_path = sys.argv[1]
    destination_path = sys.argv[2]

    reference_overrides = None
    if len(sys.argv) == 4:
        reference_overrides = load_reference_overrides_csv(sys.argv[3])

    compute_formation_enthalpy(source_path, destination_path, reference_overrides)
