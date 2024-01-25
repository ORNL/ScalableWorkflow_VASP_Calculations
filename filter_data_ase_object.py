import os
from os import listdir
from os.path import isfile, join
import torch
from torch import tensor
from torch_geometric.data import Data
import shutil
import numpy as np

from ase.io.cfg import read_cfg

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

    ase_object = read_cfg(filepath)

    data_object = Data()

    data_object.supercell_size = tensor(ase_object.cell.array).float()
    data_object.pos = tensor(ase_object.arrays["positions"]).float()
    proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
    masses = np.expand_dims(ase_object.arrays["masses"], axis=1)
    c_peratom = np.expand_dims(ase_object.arrays["c_peratom"], axis=1)
    fx = np.expand_dims(ase_object.arrays["fx"], axis=1)
    fy = np.expand_dims(ase_object.arrays["fy"], axis=1)
    fz = np.expand_dims(ase_object.arrays["fz"], axis=1)
    node_feature_matrix = np.concatenate(
        (proton_numbers, masses, c_peratom, fx, fy, fz), axis=1
    )
    data_object.x = tensor(node_feature_matrix).float()

    filename_without_extension = os.path.splitext(filepath)[0]

    if os.path.exists(os.path.join(filename_without_extension + ".bulk")):
        filename_bulk = os.path.join(filename_without_extension + ".bulk")
        f = open(filename_bulk, "r", encoding="utf-8")
        lines = f.readlines()
        graph_feat = lines[0].split(None, 2)
        g_feature = []
        # collect graph features
        for item in range(len(graph_feature_dim)):
            for icomp in range(graph_feature_dim[item]):
                it_comp = graph_feature_col[item] + icomp
                g_feature.append(float(graph_feat[it_comp].strip()))
        data_object.y = tensor(g_feature)

    return data_object


def filter_data(source_path, destination_path):

    dirs = None
    if rank == 0:
        # create a new directory for filtered data
        os.makedirs(destination_path, exist_ok=False)
        dirs = [f.name for f in os.scandir(source_path) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), size))[rank]

    for dir in sorted(dirs)[rx.start:rx.stop]:
        print("f Rank: ", rank, " - dir: ", dir, flush=True)
        os.makedirs(destination_path + '/' + dir, exist_ok=False)
        for subdir, dirs, files in os.walk(source_path + '/' + dir):
            for filename in files:
                if 'u.cfg' in filename:
                    try:
                        transform_ASE_object_to_data_object(source_path + '/' + dir + '/' + filename)
                        shutil.copy(source_path + '/' + dir + '/' + filename, destination_path + '/' + dir)
                    except:
                        print(source_path + '/' + dir + '/' + filename, "could not be converted in torch_geometric.data "
                                                                "object")


if __name__ == '__main__':
    source_path = './atoms256'
    destination_path = './atoms256_filtered'
    filter_data(source_path, destination_path)
