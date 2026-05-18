#!/usr/bin/env python3
import os
import re
import subprocess

WORKDIR = "/lustre/orion/lrn070/world-shared/mlupopa/ScalableWorkflow_VASP_Calculations"
DATASET_BASE = "/lustre/orion/lrn070/world-shared/mlupopa/RHEAs_HydraGNN/HydraGNN/examples/vasp_solid_solution_alloys/dataset"
RUN_BASE = os.path.join(WORKDIR, "corrected_DFT_pure_element_runs")


def find_outcar(directory):
    if not os.path.isdir(directory):
        return None
    best = None
    best_idx = -1
    for f in os.listdir(directory):
        m = re.match(r"^N(\\d+)\\.OUTCAR$", f)
        if m:
            idx = int(m.group(1))
            if idx > best_idx:
                best_idx = idx
                best = f
    if best is not None:
        return os.path.join(directory, best)
    plain = os.path.join(directory, "OUTCAR")
    if os.path.isfile(plain):
        return plain
    return None


def sigma0_e_per_atom(outcar_path, natoms=128):
    cmd = (
        "grep -n 'energy(sigma->0) =' '"
        + outcar_path
        + "' | tail -1 | rev | cut -d ' ' -f1 | rev"
    )
    s = subprocess.getoutput(cmd).strip()
    if not s:
        raise RuntimeError(f"No energy(sigma->0) line in {outcar_path}")
    return float(s) / natoms


def ensure_titel_token(outcar_path, token):
    titel = subprocess.getoutput(f"grep -m 6 'TITEL' '{outcar_path}'")
    if token not in titel:
        raise RuntimeError(
            f"Expected TITEL token '{token}' not found in {outcar_path}.\n"
            f"TITEL lines:\n{titel}"
        )


def write_csv(path, rows):
    with open(path, "w", encoding="ascii") as f:
        f.write("# element,eV_per_atom\n")
        for el, val in rows:
            f.write(f"{el},{val:.8f}\n")


def main():
    # New corrected runs (must exist after you run VASP in corrected_DFT_pure_element_runs/*)
    zr_nbzr_out = find_outcar(os.path.join(RUN_BASE, "bcc_NbZr_Zr128"))
    zr_tazr_out = find_outcar(os.path.join(RUN_BASE, "bcc_TaZr_Zr128"))
    zr_vzr_out = find_outcar(os.path.join(RUN_BASE, "bcc_VZr_Zr128"))
    v_vzr_out = find_outcar(os.path.join(RUN_BASE, "bcc_VZr_V128"))

    required = {
        "bcc_NbZr_Zr128": zr_nbzr_out,
        "bcc_TaZr_Zr128": zr_tazr_out,
        "bcc_VZr_Zr128": zr_vzr_out,
        "bcc_VZr_V128": v_vzr_out,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(
            "Missing OUTCAR in corrected run folders: " + ", ".join(missing)
        )

    # Validate TITEL identity for newly rerun references
    ensure_titel_token(zr_nbzr_out, "Zr_sv")
    ensure_titel_token(zr_tazr_out, "Zr_sv")
    ensure_titel_token(zr_vzr_out, "Zr_sv")
    ensure_titel_token(v_vzr_out, "V_sv")

    # Reuse already consistent pure references for Nb and Ta from original dataset
    nb_out = find_outcar(os.path.join(DATASET_BASE, "bcc_NbZr", "Nb", "Nb128", "case-1"))
    ta_out = find_outcar(os.path.join(DATASET_BASE, "bcc_TaZr", "Ta", "Ta128", "case-1"))
    if nb_out is None or ta_out is None:
        raise RuntimeError("Could not find Nb/Ta pure OUTCAR in original dataset")
    ensure_titel_token(nb_out, "Nb_sv")
    ensure_titel_token(ta_out, "Ta_pv")

    nb = sigma0_e_per_atom(nb_out)
    ta = sigma0_e_per_atom(ta_out)
    zr_nbzr = sigma0_e_per_atom(zr_nbzr_out)
    zr_tazr = sigma0_e_per_atom(zr_tazr_out)
    zr_vzr = sigma0_e_per_atom(zr_vzr_out)
    v = sigma0_e_per_atom(v_vzr_out)

    out_nbzr = os.path.join(WORKDIR, "ref_overrides_bcc_NbZr_NEW.csv")
    out_tazr = os.path.join(WORKDIR, "ref_overrides_bcc_TaZr_NEW.csv")
    out_vzr = os.path.join(WORKDIR, "ref_overrides_bcc_VZr_NEW.csv")

    write_csv(out_nbzr, [("Nb", nb), ("Zr", zr_nbzr)])
    write_csv(out_tazr, [("Ta", ta), ("Zr", zr_tazr)])
    write_csv(out_vzr, [("V", v), ("Zr", zr_vzr)])

    print("Wrote:")
    print("  ", out_nbzr)
    print("  ", out_tazr)
    print("  ", out_vzr)
    print("Values (eV/atom):")
    print(f"  Nb={nb:.8f}, Ta={ta:.8f}, V={v:.8f}")
    print(f"  Zr@NbZr={zr_nbzr:.8f}, Zr@TaZr={zr_tazr:.8f}, Zr@VZr={zr_vzr:.8f}")


if __name__ == "__main__":
    main()
