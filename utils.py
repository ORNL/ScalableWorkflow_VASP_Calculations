import os
import re

import numpy as np  # summation

import torch
from torch import tensor
from torch_geometric.data import Data

from ase.io.vasp import read_vasp_out

def flatten(l):
    return [item for sublist in l for item in sublist]

def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def extract_atom_species(outcar_path):
    # Read the OUTCAR file using ASE's read_vasp_outcar function
    outcar = read_vasp_out(outcar_path)

    # Extract atomic positions and forces for each atom
    torch_atom_numbers = torch.tensor(outcar.numbers).unsqueeze(1)

    return torch_atom_numbers


def extract_supercell(section):

    # Define the pattern to match the direct lattice vectors
    lattice_pattern = re.compile(r'\s*([-\d.]+\s+[-\d.]+\s+[-\d.]+)\s+([-\d.]+\s+[-\d.]+\s+[-\d.]+)', re.MULTILINE)

    direct_lattice_matrix = []

    # Iterate through lines in the subsection
    for line in section:
        match_supercell = lattice_pattern.match(line)
        # Extract the matched group
        if match_supercell:
            lattice_vectors = match_supercell.group(1).strip().split()
            # Convert the extracted values to floats
            lattice_vectors = list(map(float, lattice_vectors))

            # Reshape the list into a 3x3 matrix
            direct_lattice_matrix.extend([lattice_vectors[i:i + 3] for i in range(0, len(lattice_vectors), 3)])

    # I need to exclude the length vector
    direct_lattice_matrix.pop()

    return torch.tensor(direct_lattice_matrix)


def extract_positions_forces_energy(section):
    # Define regular expression patterns for POSITION and TOTAL-FORCE
    pos_force_pattern = re.compile(r'\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')
    energy_pattern = re.compile(r'\s+energy\s+without\s+entropy=\s+(-?\d+\.\d+)\s+energy\(sigma->0\) =\s+(-?\d+\.\d+)')

    # Initialize lists to store POSITION and TOTAL-FORCE
    positions_list = []
    forces_list = []
    energy = []

    # Iterate through lines in the subsection
    for line in section:
        match_pos_force = pos_force_pattern.match(line)
        match_energy = energy_pattern.match(line)
        if match_pos_force:
            # Extract values and convert to float
            position_values = [float(match_pos_force.group(i)) for i in range(1, 4)]
            force_values = [float(match_pos_force.group(i)) for i in range(4, 7)]

            # Append to lists
            positions_list.append(position_values)
            forces_list.append(force_values)

        if match_energy:
            # Extract values and convert to float
            # Define the regular expression pattern to match floating-point numbers
            pattern = re.compile(r'-?\d+\.\d+')

            # Find all matches in the input string
            matches = pattern.findall(line)

            # Extract the last match as a float
            if matches:
                last_float = float(matches[-1])
                energy = last_float
            else:
                print("No floating-point number found.")

    # Convert lists to PyTorch tensors
    positions_tensor = torch.tensor(positions_list)
    forces_tensor = torch.tensor(forces_list)
    energy_tensor = torch.tensor([energy])/positions_tensor.shape[0]

    return positions_tensor, forces_tensor, energy_tensor


def read_sections_between(file_path, start_marker, end_marker):
    sections = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            in_section = False
            current_section = []

            for line in lines:
                # Check if the current line contains the start marker
                if start_marker in line:
                    in_section = True
                    current_section = []
                    current_section.append(line)

                # Check if the current line contains the end marker
                elif (end_marker in line) and in_section:
                    in_section = False
                    current_section.append(line)
                    sections.append(current_section)

                # If we're in a section, append the line to the current section
                elif in_section:
                    current_section.append(line)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return sections


def read_outcar(file_path, extract_only_optimized_geometries = True):
    # Replace these with your file path, start marker, and end marker
    supercell_start_marker = 'VOLUME and BASIS-vectors are now :'
    supercell_end_marker = 'FORCES acting on ions'
    atomic_structure_start_marker = 'POSITION                                       TOTAL-FORCE (eV/Angst)'
    atomic_structure_end_marker = 'POTLOK'

    dataset = []

    full_string = file_path
    filename = full_string.split("/")[-1]

    # Read sections between specified markers
    result_supercell = read_sections_between(file_path, supercell_start_marker,
                                                             supercell_end_marker)

    # Read sections between specified markers
    result_atomic_structure_sections = read_sections_between(file_path, atomic_structure_start_marker, atomic_structure_end_marker)

    # Extract POSITION and TOTAL-FORCE from each section
    for i, (supercell_section, atomic_structure_section) in enumerate(zip(result_supercell, result_atomic_structure_sections), start=1):

        # Extract POSITION and TOTAL-FORCE into PyTorch tensors
        supercell = extract_supercell(supercell_section)
        positions, forces, energy = extract_positions_forces_energy(atomic_structure_section)

        data_object = Data()

        data_object.pos = positions
        data_object.supercell_size = supercell
        data_object.forces = forces
        data_object.energy = energy
        data_object.y = energy
        atom_numbers = extract_atom_species(file_path)
        data_object.atom_numbers = atom_numbers
        data_object.x = torch.cat((atom_numbers, positions, forces), dim=1)

        dataset.append(data_object)

    #plot_forces(filename, dataset)

    if extract_only_optimized_geometries:
        dataset = dataset[-1:]

    return dataset, atom_numbers.flatten().tolist()


def replace_total_energy_with_formation_energy(data_object, total_energies_pure_elements):

    count_occurrencies_atom_elements = torch.bincount(data_object.x[:,0].int(), minlength=max(list(total_energies_pure_elements.keys()))+1)
    assert torch.sum(count_occurrencies_atom_elements) == data_object.num_nodes , "number of atoms in data structure does not correspond to sum of total occurrencies of individual atom species"

    count_occurrencies_atom_elements = count_occurrencies_atom_elements / data_object.num_nodes

    for element in total_energies_pure_elements.keys():
        data_object.y = data_object.y - total_energies_pure_elements[element] * count_occurrencies_atom_elements[element].item()

    return data_object


def transform_ASE_object_to_data_object(filepath):
    # FIXME:
    #  this still assumes bulk modulus is specific to the CFG format.
    #  To deal with multiple files across formats, one should generalize this function
    #  by moving the reading of the .bulk file in a standalone routine.
    #  Morevoer, this approach assumes tha there is only one global feature to look at,
    #  and that this global feature is specicially retrieveable in a file with the string *bulk* inside.

    ase_object = read_vasp_out(filepath)

    data_object = Data()

    data_object.supercell_size = tensor(ase_object.cell.array).float()
    data_object.pos = tensor(ase_object.arrays["positions"]).float()
    proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
    forces = ase_object.calc.results["forces"]
    #stress = ase_object.calc.results["stress"]
    fermi_energy = ase_object.calc.eFermi
    free_energy = ase_object.calc.results["free_energy"]
    energy = ase_object.calc.results["energy"]
    node_feature_matrix = np.concatenate(
        (proton_numbers, forces), axis=1
    )
    data_object.x = tensor(node_feature_matrix).float()

    #data_object.y = tensor(energy)

    search_string = " 'reached required accuracy' "
    cmd = 'grep -n ' + search_string + filepath
    if subprocess.getoutput(cmd) == "":
        raise ValueError("calculation has not reached convergence")

    cmd = 'grep -n '+'"energy(sigma->0) =" '+ filepath + ' | tail -1 | rev | cut -d '+'" "'+' -f1 | rev'
    energy = float(subprocess.getoutput(cmd))
    data_object.y = tensor(energy)

    return data_object


def transform_ASE_object_to_data_object_with_formation_energy(filepath, ase_object):
    # FIXME:
    #  this still assumes bulk modulus is specific to the CFG format.
    #  To deal with multiple files across formats, one should generalize this function
    #  by moving the reading of the .bulk file in a standalone routine.
    #  Morevoer, this approach assumes tha there is only one global feature to look at,
    #  and that this global feature is specicially retrieveable in a file with the string *bulk* inside.

    data_object = Data()

    data_object.supercell_size = tensor(ase_object.cell.array).float()
    data_object.pos = tensor(ase_object.arrays["positions"]).float()
    proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
    forces = ase_object.calc.results["forces"]
    stress = ase_object.calc.results["stress"]
    fermi_energy = ase_object.calc.eFermi
    free_energy = ase_object.calc.results["free_energy"]
    energy = ase_object.calc.results["energy"]
    node_feature_matrix = np.concatenate(
        (proton_numbers, forces), axis=1
    )
    data_object.x = tensor(node_feature_matrix).float()

    formation_energy_file = open(filepath + 'formation_energy.txt', 'r')
    Lines = formation_energy_file.readlines()

    # Strips the newline character
    for line in Lines:
        data_object.y = tensor([float(line.strip())])
        if data_object.y * 1000/128 > 70.0:
            print(filepath)

    return data_object


def transform_VASP_output_to_data_object_with_formation_energy(filepath):
    """Transforms lines of strings read from the raw data EAM file to Data object and returns it.

    Parameters
    ----------
    lines:
      content of data file with all the graph information
    Returns
    ----------
    Data
        Data object representing structure of a graph sample.
    """

    if ("OUTCAR" in filepath) and ("0.OUTCAR" not in filepath):

        ase_object = read_vasp_out(filepath)

        dirpath = filepath.split("OUTCAR")[0]
        data_object = transform_ASE_object_to_data_object_with_formation_energy(dirpath, ase_object)

        return data_object

    else:
        return None


def load_raw_data(raw_data_path, comm, formation_energy=False):
    """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    After that the serialized data is stored to the serialized_dataset directory.
    """

    size = comm.Get_size()
    rank = comm.Get_rank()

    dataset = []

    dirs = None
    if rank == 0:
        dirs = [f.name for f in os.scandir(raw_data_path) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), size))[rank]

    for name in sorted(dirs)[rx.start:rx.stop]:
        print("f Rank: ", rank, " - name: ", name, flush=True)
        if name == ".DS_Store":
            continue
        if os.path.isdir(os.path.join(raw_data_path, name)):
            if name == ".DS_Store":
                continue
            dir_name = os.path.join(raw_data_path, name)
            for subname in os.listdir(dir_name):
                if subname == ".DS_Store":
                    continue
                subdir_name = os.path.join(dir_name, subname)
                for subsubname in os.listdir(subdir_name):
                    subsubdir_name = os.path.join(dir_name, subname)
                    if os.path.isfile(os.path.join(subdir_name, subsubname)):
                        data_object = None
                        try:
                            if not formation_energy:
                                data_object = transform_ASE_object_to_data_object(
                                    filepath=os.path.join(subdir_name, subsubname)
                                )
                            else:
                                data_object = transform_ASE_object_to_data_object_with_formation_energy(
                                    filepath=os.path.join(subdir_name, subsubname)
                                )
                        except:
                            pass
                        if not isinstance(data_object, type(None)):
                            dataset.append(data_object)

    return dataset


def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm



