import os
from os import listdir
from os.path import isfile, join
import subprocess

import torch
from torch import tensor
from torch_geometric.data import Data
import shutil
import numpy as np

from ase.io.vasp import read_vasp_out

from utils import nsplit

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

    ase_object = read_vasp_out(filepath, parallel=False)

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

    #data_object.y = tensor(energy)

    search_string = " 'reached required accuracy' "
    cmd = 'grep -n ' + search_string + filepath
    if subprocess.getoutput(cmd) == "":
        raise ValueError("calculation has not reached convergence")

    cmd = 'grep -n '+'"energy(sigma->0) =" '+ filepath + ' | tail -1 | rev | cut -d '+'" "'+' -f1 | rev'
    energy = float(subprocess.getoutput(cmd))
    data_object.y = tensor(energy)

    return data_object


def replace_total_energy_with_formation_energy(data_object, total_energies_pure_elements):

    count_occurrencies_atom_elements = torch.bincount(data_object.x[:,0].int(), minlength=max(list(total_energies_pure_elements.keys()))+1)
    assert torch.sum(count_occurrencies_atom_elements) == data_object.num_nodes , "number of atoms in data structure does not correspond to sum of total occurrencies of individual atom species"

    count_occurrencies_atom_elements = count_occurrencies_atom_elements / data_object.num_nodes

    for element in total_energies_pure_elements.keys():
        data_object.y = data_object.y - total_energies_pure_elements[element] * count_occurrencies_atom_elements[element].item()

    return data_object


def compute_formation_enthalpy(source_path, destination_path):
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

        Nb_formation_energy = open(destination_path + '/Nb/Nb128/case-1/'+ "formation_energy.txt", "w")
        Nb_formation_energy.write(str(0.0))
        Nb_formation_energy.write("\n")
        Nb_formation_energy.close()

        Ta_formation_energy = open(destination_path + '/Ta/Ta128/case-1/'+ "formation_energy.txt", "w")
        Ta_formation_energy.write(str(0.0))
        Ta_formation_energy.write("\n")
        Ta_formation_energy.close()

        V_formation_energy = open(destination_path + '/V/V128/case-1/'+ "formation_energy.txt", "w")
        V_formation_energy.write(str(0.0))
        V_formation_energy.write("\n")
        V_formation_energy.close()

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
        os.makedirs(destination_path + '/' + dir, exist_ok=False)
        for _, subdirs, _ in os.walk(source_path + '/' + dir):
            for subdir in subdirs:
                os.makedirs(destination_path + '/' + dir + '/'+subdir, exist_ok=False)
                for _, subsubdirs, _ in os.walk(source_path + '/' + dir + '/' + subdir):
                    for subsubdir in subsubdirs:
                        os.makedirs(destination_path + '/' + dir + '/' + subdir + '/' + subsubdir, exist_ok=False)
                        for _, _, files in os.walk(source_path + '/' + dir + '/' + subdir + '/' + subsubdir):
                            found_outcar = False
                            for file in files:
                                if file == "OUTCAR":
                                    found_outcar = True
                                    try:
                                        data_object = transform_ASE_object_to_data_object(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file)
                                        data_object = replace_total_energy_with_formation_energy(data_object, total_energies_pure_elements)
                                        shutil.copy(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file, destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir)
                                        formation_energy_file = open(destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir + '/' + "formation_energy.txt", "w")
                                        formation_energy_file.write(str(data_object.y.item()))
                                        formation_energy_file.write("\n")
                                        formation_energy_file.close()
                                    except:
                                        print(source_path + '/' + dir + '/' + subdir + '/' + subsubdir + '/' + file, "could not be converted in torch_geometric.data "
                                                                            "object")
                            # If the atomic configuration has not been completed and the OUTCAR is not available, simply remove the corresponding directory from the data replica
                            if not found_outcar:
                                shutil.rmtree(destination_path+ '/' + dir + '/' + subdir + '/' + subsubdir)

            break


if __name__ == '__main__':
    source_path = './bcc'
    destination_path = './bcc_enthalpy'
    compute_formation_enthalpy(source_path, destination_path)
