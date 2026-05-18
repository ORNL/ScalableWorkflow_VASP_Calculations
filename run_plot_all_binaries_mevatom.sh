#!/usr/bin/env bash
set -euo pipefail

WORK=/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations

python3.11 "$WORK/plot_binary_from_text_outputs.py" "$WORK/bcc_enthalpy_NbZr_meVatom_20260509" Nb Zr
python3.11 "$WORK/plot_binary_from_text_outputs.py" "$WORK/bcc_enthalpy_TaZr_meVatom_20260509" Ta Zr
python3.11 "$WORK/plot_binary_from_text_outputs.py" "$WORK/bcc_enthalpy_VZr_meVatom_20260509" V Zr

echo "PLOTS_DONE_ALL"
for d in NbZr TaZr VZr; do
  p="$WORK/bcc_enthalpy_${d}_meVatom_20260509"
  echo "[$d]"
  ls "$p"/BCC_* | sed 's#^.*/##' | sort
  echo "--"
done
