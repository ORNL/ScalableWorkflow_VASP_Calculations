#!/usr/bin/env bash
set -euo pipefail

WORK=/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations
DATA=/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset
RANKS=32

cd "$WORK"

rm -rf bcc_enthalpy_NbZr_meVatom_20260509 bcc_enthalpy_TaZr_meVatom_20260509 bcc_enthalpy_VZr_meVatom_20260509

run_one() {
  local E1="$1"
  local E2="$2"
  local SRC="$3"
  local DST="$4"

  echo "=== START ${E1}-${E2} ==="
  mpirun -n "$RANKS" python3.11 "$WORK/run_compute_formation.py" "$DATA/$SRC" "$WORK/$DST"
  mpirun -n "$RANKS" python3.11 "$WORK/run_compute_rmsd.py" "$DATA/$SRC" "$WORK/$DST"
  python3.11 "$WORK/run_plots_binary.py" "$WORK/$DST" "$E1" "$E2"
  echo "=== DONE ${E1}-${E2} ==="
}

run_one Nb Zr bcc_NbZr bcc_enthalpy_NbZr_meVatom_20260509
run_one Ta Zr bcc_TaZr bcc_enthalpy_TaZr_meVatom_20260509
run_one V Zr bcc_VZr bcc_enthalpy_VZr_meVatom_20260509

echo ALL_PIPELINES_DONE
