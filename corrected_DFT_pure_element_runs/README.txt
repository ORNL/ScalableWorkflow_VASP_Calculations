corrected_DFT_pure_element_runs

This folder contains exactly the files needed to rerun the pure-element references (except POTCAR).

Subdirectories generated:
  - bcc_NbZr_Zr128 (from bcc_NbZr)
  - bcc_TaZr_Zr128 (from bcc_TaZr)
  - bcc_VZr_Zr128 (from bcc_VZr)
  - bcc_VZr_V128 (from bcc_VZr)

Each subdirectory includes:
  - INCAR
  - KPOINTS
  - POSCAR
  - run.sh
  - check_outcar.py
  - README.txt

You must provide POTCAR manually in each subdirectory.

Practical workflow to make FE physically sensible:

1) Run these four jobs (after placing POTCAR):
  - bcc_NbZr_Zr128
  - bcc_TaZr_Zr128
  - bcc_VZr_Zr128
  - bcc_VZr_V128

2) In each job folder, validate identity after completion:
  - ./check_outcar.py Zr_sv   (for the three Zr folders)
  - ./check_outcar.py V_sv    (for bcc_VZr_V128)

3) Build new override CSVs from completed runs:
  - python3.11 corrected_DFT_pure_element_runs/build_override_csvs.py

4) Recompute FE with the new references:
  - python3.11 run_compute_formation.py \
    /lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset/bcc_NbZr \
    bcc_enthalpy_NbZr_meVatom_ref_fixed \
    ref_overrides_bcc_NbZr_NEW.csv

  - python3.11 run_compute_formation.py \
    /lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset/bcc_TaZr \
    bcc_enthalpy_TaZr_meVatom_ref_fixed \
    ref_overrides_bcc_TaZr_NEW.csv

  - python3.11 run_compute_formation.py \
    /lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset/bcc_VZr \
    bcc_enthalpy_VZr_meVatom_ref_fixed \
    ref_overrides_bcc_VZr_NEW.csv
