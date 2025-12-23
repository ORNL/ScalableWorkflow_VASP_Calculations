#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:56:29 2022

@author: 7ml
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shutil

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance
from torch import tensor

from scipy.interpolate import griddata

from os import walk

import itertools

from ase.io.vasp import read_vasp_out

plt.rcParams.update({"font.size": 20})

from utils import nsplit, flatten

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

plt.rcParams.update({"font.size": 20})

pure_elements_dictionary = {'V': 23, 'Nb': 41, 'Ta': 73}

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


def plot_data(source_path, elements_list):

    for element_selected_for_plots in elements_list:

        dataset_loc = load_raw_data(source_path)
        radius = 8.0
        loop = False
        max_neighbours = 10000

        # Compute edges
        compute_edges = RadiusGraph(
            r=radius,
            loop=loop,
            max_num_neighbors=max_neighbours,
        )
        dataset_loc [:] = [compute_edges(data) for data in dataset_loc ]

        compute_edge_lengths = Distance(norm=False, cat=True)
        dataset_loc[:] = [compute_edge_lengths(data) for data in dataset_loc]

        min_edge_length = float('inf')
        max_edge_length = - float('inf')

        for data in dataset_loc:
            min_edge_length = min(min_edge_length, torch.min(data.edge_attr).item())
            max_edge_length = max(max_edge_length, torch.max(data.edge_attr).item())

        min_edge_length = comm.allreduce(min_edge_length, op=min)
        max_edge_length = comm.allreduce(max_edge_length, op=max)

        xdata_loc = [sum(data.x[:, 0] == pure_elements_dictionary[element_selected_for_plots]).item() / data.num_nodes for data in dataset_loc]
        formation_enthalpy_loc = [data.y[0].item() for data in dataset_loc]

        assert len(xdata_loc) == len(formation_enthalpy_loc)

        tuples_loc = [(composition, enthalpy) for composition, enthalpy in zip(xdata_loc,formation_enthalpy_loc)]

        tuples_all = comm.gather(tuples_loc, root=0)

        if rank == 0:

            print("Shortest edge length: ", min_edge_length)
            print("Longest edge length: ", max_edge_length)

            tuples = flatten(tuples_all)

            xdata = [elem[0] for elem in tuples]
            formation_enthalpy = [elem[1] for elem in tuples]

            # Rescale formation enthalpy to use meV/atom
            formation_enthalpy_rescaled = [item * 1000 / 128 for item in formation_enthalpy]

            # plot formation enthalpu as a function of chemical composition
            fig, ax = plt.subplots()
            hist2d_norm = getcolordensity(xdata, formation_enthalpy_rescaled)

            plt.scatter(
                xdata, formation_enthalpy_rescaled, s=8, c=hist2d_norm, vmin=0, vmax=1
            )
            plt.clim(0, 1)
            plt.colorbar()
            plt.xlabel(element_selected_for_plots+" concentration")
            plt.ylabel('Formation Energy (meV/atom)')
            plt.title(''.join(elements_list))
            ax.set_xticks([0.0, 0.5, 1.0])
            plt.draw()
            plt.tight_layout()
            plt.savefig("./BCC_enthalpy_vs_concentration_" +element_selected_for_plots + ".png", dpi=400)


def load_raw_data(raw_data_path):
    """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    After that the serialized data is stored to the serialized_dataset directory.
    """

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
        # if the directory contains file, iterate over them
        if os.path.isfile(os.path.join(raw_data_path, name)):
            data_object = transform_CFG_input_to_data_object_base(
                filepath=os.path.join(raw_data_path, name)
            )
            if not isinstance(data_object, type(None)):
                dataset.append(data_object)
        # if the directory contains subdirectories, explore their content
        elif os.path.isdir(os.path.join(raw_data_path, name)):
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
                        data_object = transform_input_to_data_object_base(
                            filepath=os.path.join(subdir_name, subsubname)
                        )
                        if not isinstance(data_object, type(None)):
                            dataset.append(data_object)

    return dataset


def transform_input_to_data_object_base(filepath):
    data_object = transform_VASP_input_to_data_object_base(
        filepath=filepath
    )
    return data_object


def transform_VASP_input_to_data_object_base(filepath):
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

    if "OUTCAR" in filepath and "0.OUTCAR" not in filepath:

        try:

            ase_object = read_vasp_out(filepath)

            dirpath = filepath.split("OUTCAR")[0]

            data_object = transform_VASP_ASE_object_to_data_object(dirpath, ase_object)

            return data_object

        except:

            print(filepath)

    else:
        return None


def transform_VASP_ASE_object_to_data_object(filepath, ase_object):
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


if __name__ == '__main__':
    elements_list = ['Nb', 'Ta', 'V']
    source_path = './bcc_enthalpy/'+'-'.join(elements_list)
    plot_data(source_path, elements_list)
