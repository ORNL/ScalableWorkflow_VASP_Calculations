# bcc_VZr_Zr128

Required files provided: INCAR, KPOINTS, POSCAR, run.sh, check_outcar.py
Missing by design: POTCAR

Expected POTCAR identity token in OUTCAR TITEL lines: Zr_sv

Run:
  1) Place the correct POTCAR in this directory.
  2) Execute: ./run.sh
  3) Validate: ./check_outcar.py Zr_sv

Notes:
  - KPOINTS and major INCAR settings are taken from dataset bcc_VZr and forced to PREC=Normal.
  - This is a 128-atom pure-Zr reference run.
  - POSCAR was intentionally reset to a sensible Zr bcc starting cell (~13.28 A box) instead of the original
    bcc_VZr/Zr/Zr128 source cell (~11.97 A), because that source pure-Zr reference is provenance-corrupted.
