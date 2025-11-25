import os, re
from os import listdir
from os.path import isfile, join
import subprocess

from torch import tensor
from torch_geometric.data import Data
import shutil
import numpy as np


import torch

def extract_coordinates(file_path):
    # Read the POSCAR file content
    with open(file_path, 'r') as file:
        poscar_content = file.read()

    # Find the index of the line with "Direct"
    start_index = poscar_content.find("Direct")

    # Extract lines after "Direct"
    coordinates_section = poscar_content[start_index + len("Direct"):].strip()

    # Split the lines and extract coordinates until an empty line is encountered
    coordinates_lines = coordinates_section.split('\n')
    coordinates_data = []

    for line in coordinates_lines:
        if not line.strip():  # Stop when an empty line is encountered
            break
        coordinates_data.append(list(map(float, line.split())))

    # Convert to torch tensor
    tensor_coordinates = torch.tensor(coordinates_data)

    return tensor_coordinates

from ase.io.vasp import read_vasp_out

from utils import nsplit, replace_total_energy_with_formation_energy

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def find_final_outcar(directory):
    """Find the OUTCAR file with the largest B in NB.OUTCAR format, or plain OUTCAR as fallback."""
    pattern = re.compile(r'^N(\d+)\.OUTCAR$')
    max_index = -1
    max_file = None
    has_plain_outcar = False
    
    try:
        files = os.listdir(directory)
        for file in files:
            # Check for plain OUTCAR
            if file == 'OUTCAR':
                has_plain_outcar = True
            # Check for N*.OUTCAR pattern
            match = pattern.match(file)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                    max_file = file
    except FileNotFoundError:
        return None
    
    # Return N*.OUTCAR if found, otherwise return plain OUTCAR if it exists
    if max_file is not None:
        return max_file
    elif has_plain_outcar:
        return 'OUTCAR'
    else:
        return None


def find_final_contcar(directory):
    """Find the CONTCAR file with the largest B in NB.CONTCAR format, or plain CONTCAR as fallback."""
    pattern = re.compile(r'^N(\d+)\.CONTCAR$')
    max_index = -1
    max_file = None
    has_plain_contcar = False
    
    try:
        files = os.listdir(directory)
        for file in files:
            # Check for plain CONTCAR
            if file == 'CONTCAR':
                has_plain_contcar = True
            # Check for N*.CONTCAR pattern
            match = pattern.match(file)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
                    max_file = file
    except FileNotFoundError:
        return None
    
    # Return N*.CONTCAR if found, otherwise return plain CONTCAR if it exists
    if max_file is not None:
        return max_file
    elif has_plain_contcar:
        return 'CONTCAR'
    else:
        return None


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



def compute_mean_squared_displacement(source_path, destination_path):
    
    # Element atomic numbers mapping (can be extended)
    element_atomic_numbers = {
        'Nb': 41, 'Ta': 73, 'V': 23, 'Zr': 40, 'Hf': 72, 'Ti': 22
    }
    
    pure_elements = []

    if rank == 0:
        # Automatically discover pure element directories
        print(f"Discovering pure element directories in {source_path}", flush=True)
        first_level_dirs = [f.name for f in os.scandir(source_path) if f.is_dir()]
        
        for first_dir in first_level_dirs:
            second_level_dirs = [f.name for f in os.scandir(source_path + '/' + first_dir) if f.is_dir()]
            # Look for directories matching pattern like Nb128, Zr128, etc. (element + 128)
            for dir_name in second_level_dirs:
                # Check if directory name ends with 128 and starts with a known element
                if dir_name.endswith('128'):
                    element = dir_name[:-3]  # Remove '128'
                    if element in element_atomic_numbers:
                        pure_elements.append({
                            'element': element,
                            'dir_name': dir_name,
                            'first_level': first_dir,
                            'atomic_number': element_atomic_numbers[element]
                        })
        
        print(f"Found pure elements: {[e['element'] for e in pure_elements]}", flush=True)
        
        # Process each pure element
        for elem_info in pure_elements:
            element = elem_info['element']
            dir_name = elem_info['dir_name']
            first_level = elem_info['first_level']
            
            rmsd_file = open(source_path + f'/{first_level}/{dir_name}/case-1/root_mean_squared_displacement.txt', "w")
            rmsd_file.write(str(0.0))
            rmsd_file.write("\n")
            rmsd_file.close()
            print(f"Created RMSD file for {element}", flush=True)
    
    comm.Barrier()

    # Broadcast pure element data to all ranks
    pure_elements = comm.bcast(pure_elements, root=0)

    dirs = None

    if rank == 0:
        # First, find all subdirectories at source_path level
        first_level_dirs = [f.name for f in os.scandir(source_path) if f.is_dir()]
        
        # For each first-level directory, get the actual data directories (second level)
        dirs = []
        for first_dir in first_level_dirs:
            second_level_dirs = [f.name for f in os.scandir(source_path + '/' + first_dir) if f.is_dir()]
            # Store as tuples of (first_level, second_level) to preserve the path
            for second_dir in second_level_dirs:
                dirs.append((first_dir, second_dir))
        
        # Remove pure element directories (matching Element128 pattern)
        pure_element_dir_names = [e['dir_name'] for e in pure_elements]
        dirs = [(f, s) for (f, s) in dirs if s not in pure_element_dir_names]

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), size))[rank]

    for first_dir, dir in sorted(dirs)[rx.start:rx.stop]:
        print(f"Rank: {rank} - Processing dir: {first_dir}/{dir}", flush=True)
        for _, subdirs, _ in os.walk(source_path + '/' + first_dir + '/' + dir):
            total_cases = len(subdirs)
            print(f"Rank: {rank} - Found {total_cases} cases in {dir}", flush=True)
            
            for idx, subdir in enumerate(subdirs, 1):
                # Find the final OUTCAR file with largest B in NB.OUTCAR
                final_outcar = find_final_outcar(source_path + '/' + first_dir + '/' + dir + '/' + subdir)
                
                if final_outcar:
                    try:
                        print(f"Rank: {rank} - Processing {dir}/{subdir} ({idx}/{total_cases}) - Reading {final_outcar}", flush=True)
                        data_object = transform_ASE_object_to_data_object(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + final_outcar)
                        
                        # Create destination directory if it doesn't exist
                        dest_dir = destination_path + '/' + dir + '/' + subdir
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        # Find the corresponding CONTCAR file
                        final_contcar = find_final_contcar(source_path + '/' + first_dir + '/' + dir + '/' + subdir)
                        if not final_contcar:
                            raise FileNotFoundError(f"No CONTCAR file found in {source_path}/{first_dir}/{dir}/{subdir}")
                        
                        if "-bis" in final_outcar:
                            initial_ideal_bcc = extract_coordinates(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + '0.POSCAR-bis')
                            final_bcc = extract_coordinates(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + final_contcar)
                            root_mean_squared_displacement_file = open(
                                dest_dir + '/' + "root_mean_squared_displacement-bis.txt", "w")
                        else:
                            initial_ideal_bcc = extract_coordinates(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + '0.POSCAR')
                            final_bcc = extract_coordinates(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + final_contcar)
                            root_mean_squared_displacement_file = open(
                                dest_dir + '/' + "root_mean_squared_displacement.txt", "w")
                        
                        atomic_displacements = final_bcc - initial_ideal_bcc
                        atomic_displacements[atomic_displacements > 0.9] -= 1
                        atomic_displacements[atomic_displacements < -0.9] += 1
                        distorted_atomic_displacement = torch.matmul(data_object.supercell_size, atomic_displacements.t())
                        norm_distorted_atomic_displacement = torch.norm(distorted_atomic_displacement.t(), dim=1)**2
                        mean_squared_displacement = torch.sum(norm_distorted_atomic_displacement)/norm_distorted_atomic_displacement.shape[0]
                        root_mean_squared_displacement = torch.sqrt(mean_squared_displacement)
                        root_mean_squared_displacement_file.write(str(root_mean_squared_displacement.item()))
                        root_mean_squared_displacement_file.write("\n")
                        root_mean_squared_displacement_file.close()
                        print(f"Rank: {rank} - Completed {dir}/{subdir} ({idx}/{total_cases})", flush=True)
                    except Exception as e:
                        print(f"Rank: {rank} - ERROR: {source_path}/{first_dir}/{dir}/{subdir}/{final_outcar} could not be processed: {e}", flush=True)
                else:
                    print(f"Rank: {rank} - WARNING: No OUTCAR found in {dir}/{subdir}", flush=True)

            break


if __name__ == '__main__':
    source_path = '../bcc_Nb-Zr'
    destination_path = './bcc_enthalpy_NbZr'
    compute_mean_squared_displacement(source_path, destination_path)

