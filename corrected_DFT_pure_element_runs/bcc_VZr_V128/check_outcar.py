#!/usr/bin/env python3
import re,subprocess,sys

outcar='OUTCAR'
expected=sys.argv[1]

try:
    titel=subprocess.getoutput("grep -m 4 'TITEL' OUTCAR")
except Exception:
    print('Could not read OUTCAR TITEL lines')
    sys.exit(2)

print('TITEL lines:')
print(titel)
if expected not in titel:
    print(f'ERROR: expected potential token {expected} not found in TITEL lines')
    sys.exit(1)

# print sigma->0 energy for convenience
es=subprocess.getoutput("grep -n 'energy(sigma->0) =' OUTCAR | tail -1 | rev | cut -d ' ' -f1 | rev").strip()
if es:
    try:
        e=float(es)
        print(f'energy(sigma->0) total eV: {e:.10f}')
        print(f'energy(sigma->0) eV/atom (128): {e/128.0:.10f}')
    except Exception:
        pass
print('OK')
