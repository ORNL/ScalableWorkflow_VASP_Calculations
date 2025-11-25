import os
from os import listdir
from os.path import isfile, join
import subprocess
import re

import torch
from torch import tensor
from torch_geometric.data import Data
import shutil
import numpy as np

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


def transform_ASE_object_to_data_object(filepath):

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

    search_string = " 'reached required accuracy' "
    cmd = 'grep -n ' + search_string + filepath
    if subprocess.getoutput(cmd) == "":
        raise ValueError("calculation has not reached convergence")

    cmd = 'grep -n '+'"energy(sigma->0) =" '+ filepath + ' | tail -1 | rev | cut -d '+'" "'+' -f1 | rev'
    energy = float(subprocess.getoutput(cmd))

    # Convert energy from eV to meV
    # Normalize energy by the number of atoms in the structure
    data_object.y = tensor(energy) * 1000/data_object.num_nodes

    return data_object



def compute_formation_enthalpy(source_path, destination_path):
    
    # Element atomic numbers mapping (can be extended)
    element_atomic_numbers = {
        'Nb': 41, 'Ta': 73, 'V': 23, 'Zr': 40, 'Hf': 72, 'Ti': 22
    }
    
    total_energies_pure_elements = {}
    pure_element_data = {}

    if rank == 0:
        # Create base directory
        print(f"Creating directories in {destination_path}", flush=True)
        os.makedirs(destination_path, exist_ok=False)
        
        # Automatically discover pure element directories
        print(f"Discovering pure element directories in {source_path}", flush=True)
        first_level_dirs = [f.name for f in os.scandir(source_path) if f.is_dir()]
        
        pure_elements = []
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
            atomic_num = elem_info['atomic_number']
            
            # Create output directories
            os.makedirs(destination_path + f'/{element}', exist_ok=False)
            os.makedirs(destination_path + f'/{element}/{dir_name}', exist_ok=False)
            os.makedirs(destination_path + f'/{element}/{dir_name}/case-1', exist_ok=False)
            
            # Find and read OUTCAR
            outcar_path = source_path + f'/{first_level}/{dir_name}/case-1'
            final_outcar = find_final_outcar(outcar_path)
            
            if final_outcar is None:
                raise FileNotFoundError(f"Could not find OUTCAR for {element} in {outcar_path}")
            
            print(f"Reading {element} OUTCAR: {outcar_path}/{final_outcar}", flush=True)
            pure_object = transform_ASE_object_to_data_object(outcar_path + '/' + final_outcar)
            
            # Copy OUTCAR and write formation energy
            shutil.copy(outcar_path + '/' + final_outcar, 
                       destination_path + f'/{element}/{dir_name}/case-1/' + final_outcar)
            
            formation_energy_file = open(destination_path + f'/{element}/{dir_name}/case-1/formation_energy.txt', "w")
            formation_energy_file.write(str(0.0))
            formation_energy_file.write("\n")
            formation_energy_file.close()
            
            # Store energy
            pure_element_data[element] = {
                'energy': pure_object.y.item(),
                'atomic_number': atomic_num,
                'dir_name': dir_name
            }
            total_energies_pure_elements[atomic_num] = pure_object.y.item()
            
            print(f"Successfully processed {element} with energy {pure_object.y.item()}", flush=True)
    
    comm.Barrier()

    # Broadcast pure element data to all ranks
    total_energies_pure_elements = comm.bcast(total_energies_pure_elements, root=0)
    pure_element_data = comm.bcast(pure_element_data, root=0)

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
        os.makedirs(destination_path + '/' + dir, exist_ok=False)
        for _, subdirs, _ in os.walk(source_path + '/' + first_dir + '/' + dir):
            total_cases = len(subdirs)
            print(f"Rank: {rank} - Found {total_cases} cases in {dir}", flush=True)
            
            for idx, subdir in enumerate(subdirs, 1):
                os.makedirs(destination_path + '/' + dir + '/'+subdir, exist_ok=False)
                # Find the final OUTCAR file with largest B in NB.OUTCAR
                final_outcar = find_final_outcar(source_path + '/' + first_dir + '/' + dir + '/' + subdir)
                
                if final_outcar:
                    try:
                        print(f"Rank: {rank} - Processing {dir}/{subdir} ({idx}/{total_cases}) - Reading {final_outcar}", flush=True)
                        data_object = transform_ASE_object_to_data_object(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + final_outcar)
                        data_object = replace_total_energy_with_formation_energy(data_object, total_energies_pure_elements)
                        shutil.copy(source_path + '/' + first_dir + '/' + dir + '/' + subdir + '/' + final_outcar, destination_path+ '/' + dir + '/' + subdir)
                        
                        # Convert to formation energy per atom
                        formation_energy_per_atom = data_object.y.item() / data_object.num_nodes
                        
                        if "-bis" in final_outcar:
                            formation_energy_file = open(destination_path+ '/' + dir + '/' + subdir + '/' + "formation_energy-bis.txt", "w")
                        else:
                            formation_energy_file = open(destination_path + '/' + dir + '/' + subdir + '/' + "formation_energy.txt","w")
                        formation_energy_file.write(str(formation_energy_per_atom))
                        formation_energy_file.write("\n")
                        formation_energy_file.close()
                        print(f"Rank: {rank} - Completed {dir}/{subdir} ({idx}/{total_cases})", flush=True)
                    except Exception as e:
                        print(f"Rank: {rank} - ERROR: {source_path}/{first_dir}/{dir}/{subdir}/{final_outcar} could not be converted: {e}", flush=True)
                else:
                    print(f"Rank: {rank} - WARNING: No OUTCAR found in {dir}/{subdir}, removing directory", flush=True)
                    # If the atomic configuration has not been completed and the OUTCAR is not available, simply remove the corresponding directory from the data replica
                    shutil.rmtree(destination_path+ '/' + dir + '/' + subdir)

            break


if __name__ == '__main__':
    source_path = '../bcc_Nb-Zr'
    destination_path = './bcc_enthalpy_NbZr'
    compute_formation_enthalpy(source_path, destination_path)
