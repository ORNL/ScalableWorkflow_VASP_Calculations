# bcc_VZr_V128

Required files provided: INCAR, KPOINTS, POSCAR, run.sh, check_outcar.py
Missing by design: POTCAR

Expected POTCAR identity token in OUTCAR TITEL lines: V_sv

Run:
  1) Place the correct POTCAR in this directory.
  2) Execute: ./run.sh
  3) Validate: ./check_outcar.py V_sv

Notes:
  - KPOINTS and major INCAR settings are taken from dataset bcc_VZr and forced to PREC=Normal.
  - This is a 128-atom pure-V reference run.
