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


def extract_direct_lattice_vectors_from_contcar(contcar_path):
    # Read the CONTCAR file
    with open(contcar_path, 'r') as file:
        lines = file.readlines()

    # The lattice vectors are usually located on lines 3, 4, and 5
    lattice_vectors = []

    for i in range(2, 5):
        line = lines[i].strip()
        vector = [float(val) for val in line.split()]
        lattice_vectors.append(vector)

    return torch.tensor(lattice_vectors)

def extract_last_direct_lattice_vectors_from_outcar(outcar_path):
    # Read the OUTCAR file
    with open(outcar_path, 'r') as file:
        lines = file.readlines()

    # Initialize variables to hold the direct lattice vectors
    lattice_vectors = []

    # Loop through the file lines in reverse to find the last occurrence of "direct lattice vectors"
    for line in reversed(lines):
        if "direct lattice vectors" in line:
            # The vectors are the next three lines after "direct lattice vectors"
            idx = lines.index(line) + 1
            lattice_vectors = [lines[idx].strip(),
                               lines[idx + 1].strip(),
                               lines[idx + 2].strip()]
            break

    # If found, convert strings to lists of floats
    if lattice_vectors:
        lattice_vectors = [[float(val) for val in vector.split()] for vector in lattice_vectors]

    return torch.tensor(lattice_vectors)

from ase.io.vasp import read_vasp_out

from utils import nsplit, replace_total_energy_with_formation_energy

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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
    total_energies_pure_elements = {23: 0.0, 41: 0.0, 73: 0.0}

    pure_Nb_total_energy = 0.0
    pure_Ta_total_energy = 0.0
    pure_V_total_energy = 0.0

    if rank == 0:

        # create a new directory for filtered data
        """
        os.makedirs(destination_path, exist_ok=False)
        os.makedirs(destination_path + '/Nb', exist_ok=False)
        os.makedirs(destination_path + '/Nb/Nb128', exist_ok=False)
        os.makedirs(destination_path + '/Nb/Nb128/case-1', exist_ok=False)
        os.makedirs(destination_path + '/Ta', exist_ok=False)
        os.makedirs(destination_path + '/Ta/Ta128', exist_ok=False)
        os.makedirs(destination_path + '/Ta/Ta128/case-1', exist_ok=False)
        os.makedirs(destination_path + '/V', exist_ok=False)
        os.makedirs(destination_path + '/V/V128', exist_ok=False)
        os.makedirs(destination_path + '/V/V128/case-1', exist_ok=False)

        shutil.copy(source_path + '/Nb/Nb128/case-1/OUTCAR', destination_path + '/Nb/Nb128/case-1/OUTCAR')
        shutil.copy(source_path + '/Ta/Ta128/case-1/OUTCAR', destination_path + '/Ta/Ta128/case-1/OUTCAR')
        shutil.copy(source_path + '/V/V128/case-1/OUTCAR', destination_path + '/V/V128/case-1/OUTCAR')
        """

        tensor_zeros = torch.zeros(3, 3)
        tensor_zeros_str = '\n'.join([' '.join(map(str, row.tolist())) for row in tensor_zeros])

        #Nb_lattice_vectors = open(destination_path + '/Nb/Nb128/case-1/'+ "mean_squared_displacement.txt", "w")
        Nb_lattice_vectors = open(source_path + '/Nb/Nb128/case-1/' + "deformation_lattice_vectors.txt", "w")
        Nb_lattice_vectors.write(tensor_zeros_str)
        Nb_lattice_vectors.write("\n")
        Nb_lattice_vectors.close()

        #Ta_lattice_vectors = open(destination_path + '/Ta/Ta128/case-1/'+ "mean_squared_displacement.txt", "w")
        Ta_lattice_vectors = open(source_path + '/Ta/Ta128/case-1/' + "deformation_lattice_vectors.txt", "w")
        Ta_lattice_vectors.write(tensor_zeros_str)
        Ta_lattice_vectors.write("\n")
        Ta_lattice_vectors.close()

        #V_lattice_vectors = open(destination_path + '/V/V128/case-1/'+ "mean_squared_displacement.txt", "w")
        V_lattice_vectors = open(source_path + '/V/V128/case-1/' + "deformation_lattice_vectors.txt", "w")
        V_lattice_vectors.write(tensor_zeros_str)
        V_lattice_vectors.write("\n")
        V_lattice_vectors.close()

    comm.Barrier()

    dirs = None

    if rank == 0:
        dirs = [f.name for f in os.scandir(source_path) if f.is_dir()]

    # Remove directories of pure elements from dirs
    dirs.remove('Nb')
    dirs.remove('Ta')
    dirs.remove('V')

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), size))[rank]

    for dir in sorted(dirs)[rx.start:rx.stop]:
        print("f Rank: ", rank, " - dir: ", dir, flush=True)
        #os.makedirs(destination_path + '/' + dir, exist_ok=False)
        for _, subdirs, _ in os.walk(source_path + '/' + dir):
            for subdir in subdirs:
                #os.makedirs(destination_path + '/' + dir + '/'+subdir, exist_ok=False)
                for _, subsubdirs, _ in os.walk(source_path + '/' + dir + '/' + subdir):
                    for subsubdir in subsubdirs:
                        #os.makedirs(destination_path + '/' + dir + '/' + subdir + '/' + subsubdir, exist_ok=False)
                        for _, _, files in os.walk(source_path + '/' + dir + '/' + subdir + '/' + subsubdir):
                            found_outcar = False
                            for file in files:
                                if file == "OUTCAR" or file == "OUTCAR-bis":
                                    found_outcar = True
                                    try:
                                        data_object = transform_ASE_object_to_data_object(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file)
                                        #shutil.copy(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file, destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir)
                                        if "-bis" in file:
                                            initial_lattice_vectors = extract_direct_lattice_vectors_from_contcar(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + 'POSCAR-bis')
                                            final_lattice_vectors = extract_direct_lattice_vectors_from_contcar(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + 'CONTCAR-bis')
                                            deformation_lattice_vectors_file = open(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "deformation_lattice_vectors-bis.txt",
                                                "w")
                                        else:
                                            initial_lattice_vectors = extract_direct_lattice_vectors_from_contcar(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + 'POSCAR')
                                            final_lattice_vectors = extract_direct_lattice_vectors_from_contcar(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + 'CONTCAR')
                                            deformation_lattice_vectors_file = open(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "deformation_lattice_vectors.txt",
                                                "w")                                        #distorted_ideal_bcc_lattice = torch.matmul(data_object.supercell_size, initial_ideal_bcc.t())
                                        #atomic_displacements = torch.norm(distorted_ideal_bcc_lattice.t()-data_object.pos, dim=1)
                                        deformation_lattice_vectors = final_lattice_vectors - initial_lattice_vectors
                                        tensor_str = '\n'.join([' '.join(map(str, row.tolist())) for row in deformation_lattice_vectors])
                                        deformation_lattice_vectors_file.write(tensor_str)
                                        deformation_lattice_vectors_file.write("\n")
                                        deformation_lattice_vectors_file.close()
                                    except:
                                        print(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file, "could not be converted in torch_geometric.data "
                                                                            "object")
                            # If the atomic configuration has not been completed and the OUTCAR is not available, simply remove the corresponding directory from the data replica
                            #if not found_outcar:
                            #    shutil.rmtree(destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir)

            break


if __name__ == '__main__':
    source_path = '10.13139_OLCF_2222910/bcc_Ta-V'
    destination_path = './bcc_enthalpy'
    compute_mean_squared_displacement(source_path, destination_path)

