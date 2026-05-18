#!/usr/bin/env bash
set -euo pipefail

WORK=/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations
DATA=/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset
RANKS=32

cd "$WORK"

rm -rf bcc_enthalpy_NbZr_meVatom_20260509 bcc_enthalpy_TaZr_meVatom_20260509 bcc_enthalpy_VZr_meVatom_20260509

echo "START Nb-Zr"
mpirun -n "$RANKS" python3.11 "$WORK/run_compute_formation.py" "$DATA/bcc_NbZr" "$WORK/bcc_enthalpy_NbZr_meVatom_20260509"
mpirun -n "$RANKS" python3.11 "$WORK/run_compute_rmsd.py" "$DATA/bcc_NbZr" "$WORK/bcc_enthalpy_NbZr_meVatom_20260509"
echo "DONE Nb-Zr"

echo "START Ta-Zr"
mpirun -n "$RANKS" python3.11 "$WORK/run_compute_formation.py" "$DATA/bcc_TaZr" "$WORK/bcc_enthalpy_TaZr_meVatom_20260509"
mpirun -n "$RANKS" python3.11 "$WORK/run_compute_rmsd.py" "$DATA/bcc_TaZr" "$WORK/bcc_enthalpy_TaZr_meVatom_20260509"
echo "DONE Ta-Zr"

echo "START V-Zr"
mpirun -n "$RANKS" python3.11 "$WORK/run_compute_formation.py" "$DATA/bcc_VZr" "$WORK/bcc_enthalpy_VZr_meVatom_20260509"
mpirun -n "$RANKS" python3.11 "$WORK/run_compute_rmsd.py" "$DATA/bcc_VZr" "$WORK/bcc_enthalpy_VZr_meVatom_20260509"
echo "DONE V-Zr"

echo "ALL_COMPUTE_DONE"
