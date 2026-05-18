#!/usr/bin/env bash
set -euo pipefail

WORK=/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations
DATA=/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset

run_one() {
  local tag="$1"
  local e1="$2"
  local e2="$3"
  local src="$4"
  local refcsv="$5"
  local out="$WORK/${tag}_override_20260509"

  if [[ -e "$out" ]]; then
    echo "Removing existing output: $out"
    rm -rf "$out"
  fi

  echo "=== COMPUTE $e1-$e2 ==="
  mpirun -n 32 python3.11 "$WORK/run_compute_formation.py" "$DATA/$src" "$out" "$WORK/$refcsv"

  echo "=== PLOT $e1-$e2 ==="
  python3.11 "$WORK/plot_binary_from_text_outputs.py" "$out" "$e1" "$e2"
}

run_one bcc_enthalpy_NbZr_meVatom Nb Zr bcc_NbZr ref_overrides_bcc_NbZr.csv
run_one bcc_enthalpy_TaZr_meVatom Ta Zr bcc_TaZr ref_overrides_bcc_TaZr.csv
run_one bcc_enthalpy_VZr_meVatom V Zr bcc_VZr ref_overrides_bcc_VZr.csv

echo "OVERRIDE_COMPUTE_AND_PLOT_DONE"
