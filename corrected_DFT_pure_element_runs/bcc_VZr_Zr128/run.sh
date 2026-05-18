#!/bin/bash
set -euo pipefail

# Fill in your VASP launch command here.
# Examples:
#   srun -n 128 vasp_std
#   mpirun -np 128 vasp_std

vasp_std > vasp.out 2>&1
