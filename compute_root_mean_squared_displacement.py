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

        pure_Nb_object = transform_ASE_object_to_data_object(source_path + '/Nb/Nb128/case-1/OUTCAR')
        pure_Ta_object = transform_ASE_object_to_data_object(source_path + '/Ta/Ta128/case-1/OUTCAR')
        pure_V_object = transform_ASE_object_to_data_object(source_path + '/V/V128/case-1/OUTCAR')
        shutil.copy(source_path + '/Nb/Nb128/case-1/OUTCAR', destination_path + '/Nb/Nb128/case-1/OUTCAR')
        shutil.copy(source_path + '/Ta/Ta128/case-1/OUTCAR', destination_path + '/Ta/Ta128/case-1/OUTCAR')
        shutil.copy(source_path + '/V/V128/case-1/OUTCAR', destination_path + '/V/V128/case-1/OUTCAR')

        Nb_rmsd = open(destination_path + '/Nb/Nb128/case-1/'+ "mean_squared_displacement.txt", "w")
        Nb_rmsd.write(str(0.0))
        Nb_rmsd.write("\n")
        Nb_rmsd.close()

        Ta_rmsd = open(destination_path + '/Ta/Ta128/case-1/'+ "mean_squared_displacement.txt", "w")
        Ta_rmsd.write(str(0.0))
        Ta_rmsd.write("\n")
        Ta_rmsd.close()

        V_rmsd = open(destination_path + '/V/V128/case-1/'+ "mean_squared_displacement.txt", "w")
        V_rmsd.write(str(0.0))
        V_rmsd.write("\n")
        V_rmsd.close()

        pure_Nb_total_energy = pure_Nb_object.y.item()
        pure_Ta_total_energy = pure_Ta_object.y.item()
        pure_V_total_energy = pure_V_object.y.item()

    comm.Barrier()

    pure_Nb_total_energy = comm.bcast(pure_Nb_total_energy, root=0)
    pure_Ta_total_energy = comm.bcast(pure_Ta_total_energy, root=0)
    pure_V_total_energy = comm.bcast(pure_V_total_energy, root=0)

    total_energies_pure_elements[41] = pure_Nb_total_energy
    total_energies_pure_elements[73] = pure_Ta_total_energy
    total_energies_pure_elements[23] = pure_V_total_energy

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
                                            initial_ideal_bcc = extract_coordinates(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + '0.POSCAR-bis')
                                            final_bcc = extract_coordinates(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + 'CONTCAR-bis')
                                            #formation_energy_file = open(destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "formation_energy-bis.txt", "w")
                                            root_mean_squared_displacement_file = open(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "root_mean_squared_displacement-bis.txt",
                                                "w")
                                        else:
                                            #formation_energy_file = open(destination_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "formation_energy.txt","w")
                                            initial_ideal_bcc = extract_coordinates(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + '0.POSCAR')
                                            final_bcc = extract_coordinates(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + 'CONTCAR')
                                            root_mean_squared_displacement_file = open(
                                                source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "root_mean_squared_displacement.txt",
                                                "w")
                                        #distorted_ideal_bcc_lattice = torch.matmul(data_object.supercell_size, initial_ideal_bcc.t())
                                        #atomic_displacements = torch.norm(distorted_ideal_bcc_lattice.t()-data_object.pos, dim=1)
                                        atomic_displacements = final_bcc - initial_ideal_bcc
                                        atomic_displacements[atomic_displacements > 0.9] -= 1
                                        atomic_displacements[atomic_displacements < -0.9] += 1
                                        distorted_atomic_displacement = torch.matmul(data_object.supercell_size,
                                                                                   atomic_displacements.t())
                                        norm_distorted_atomic_displacement = torch.norm(distorted_atomic_displacement.t(), dim=1)**2
                                        mean_squared_displacement = torch.sum(norm_distorted_atomic_displacement)/norm_distorted_atomic_displacement.shape[0]
                                        root_mean_squared_displacement = torch.sqrt(mean_squared_displacement)
                                        root_mean_squared_displacement_file.write(str(root_mean_squared_displacement.item()))
                                        root_mean_squared_displacement_file.write("\n")
                                        root_mean_squared_displacement_file.close()
                                    except:
                                        print(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file, "could not be converted in torch_geometric.data "
                                                                            "object")
                            # If the atomic configuration has not been completed and the OUTCAR is not available, simply remove the corresponding directory from the data replica
                            #if not found_outcar:
                            #    shutil.rmtree(destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir)

            break


if __name__ == '__main__':
    source_path = '../10.13139_OLCF_2222910/bcc'
    destination_path = './bcc_enthalpy'
    compute_mean_squared_displacement(source_path, destination_path)

