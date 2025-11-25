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

from ase.io.vasp import read_vasp_out

plt.rcParams.update({"font.size": 20})

from utils import nsplit, flatten

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

plt.rcParams.update({"font.size": 20})

pure_elements_dictionary = {'V': 23, 'Nb': 41, 'Ta': 73, 'Ti': 22, 'Zr': 40, 'Hf': 72}

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

    assert len(elements_list) == 2, "Plots are supported only for binaries"

    min_atom_number = min( pure_elements_dictionary[elements_list[0]], pure_elements_dictionary[elements_list[1]] )
    element_selected_for_plots = elements_list[0] if min_atom_number == pure_elements_dictionary[elements_list[0]] else elements_list[1]

    dataset_loc = load_raw_data(source_path)
    radius = 8.0
    loop = False
    max_neighbours = 10000

    """
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
    """

    xdata_loc = [sum(data.x[:, 0] == min_atom_number).item() / data.num_nodes for data in dataset_loc]
    msd_loc = [data.y[0].item() for data in dataset_loc]

    assert len(xdata_loc) == len(msd_loc)

    tuples_loc = [(composition, msd) for composition, msd in zip(xdata_loc,msd_loc)]

    tuples_all = comm.gather(tuples_loc, root=0)

    if rank == 0:

        #print("Shortest edge length: ", min_edge_length)
        #print("Longest edge length: ", max_edge_length)

        tuples = flatten(tuples_all)

        xdata = [elem[0] for elem in tuples]
        msd = [elem[1] for elem in tuples]

        q25_msd, q75_msd = np.percentile(msd, [25, 75])
        bin_width_msd = 2 * (q75_msd - q25_msd) * len(msd) ** (-1 / 3)
        bins_formation_enthalpy = round((max(msd) - min(msd)) / bin_width_msd)
        print("Minimum and Maximum of Root Mean Squared Displacement: ", min(msd[2:-1]), " - ", max(msd[2:-1]))
        # print("Freedman–Diaconis number of bins:", bins_formation_enthalpy)
        plt.figure()
        plt.hist(msd, color="blue", density=False, bins=100)  # density=False would make counts
        plt.ylabel('Number of configurations')
        plt.xlabel('RMSD (angstrom)')
        plt.title(elements_list[0]+elements_list[1]+' - BCC phase')
        plt.draw()
        plt.tight_layout()
        plt.savefig('BCC_MSD_Histogram_'+elements_list[0]+elements_list[1])

        # plot formation enthalpu as a function of chemical composition
        fig, ax = plt.subplots()
        hist2d_norm = getcolordensity(xdata, msd)

        plt.scatter(
            xdata, msd, s=8, c=hist2d_norm, vmin=0, vmax=1
        )
        plt.clim(0, 1)
        plt.colorbar()
        plt.xlabel(element_selected_for_plots+" concentration")
        plt.ylabel('RMSD (angstrom)')
        plt.title(elements_list[0]+elements_list[1])
        ax.set_xticks([0.0, 0.5, 1.0])
        plt.draw()
        plt.tight_layout()
        plt.savefig("./BCC_MSD_vs_concentration_" + elements_list[0]+elements_list[1] + ".png", dpi=400)


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
        print(f"Rank: {rank} - Processing directory: {name}", flush=True)
        if name == ".DS_Store":
            continue
        # if the directory contains subdirectories, explore their content
        if os.path.isdir(os.path.join(raw_data_path, name)):
            if name == ".DS_Store":
                continue
            dir_name = os.path.join(raw_data_path, name)
            subdirs = [s for s in os.listdir(dir_name) if s != ".DS_Store" and os.path.isdir(os.path.join(dir_name, s))]
            total_cases = len(subdirs)
            print(f"Rank: {rank} - Found {total_cases} cases in {name}", flush=True)
            
            for idx, subname in enumerate(subdirs, 1):
                subdir_name = os.path.join(dir_name, subname)
                    
                # Look for OUTCAR file in this case directory
                outcar_files = [f for f in os.listdir(subdir_name) if 'OUTCAR' in f and '0.OUTCAR' not in f]
                if outcar_files:
                    # Use the first OUTCAR found
                    outcar_file = outcar_files[0]
                    print(f"Rank: {rank} - Processing {name}/{subname} ({idx}/{total_cases}) - Reading {outcar_file}", flush=True)
                    filepath = os.path.join(subdir_name, outcar_file)
                    data_object = transform_input_to_data_object_base(filepath=filepath)
                    if not isinstance(data_object, type(None)):
                        dataset.append(data_object)
                        print(f"Rank: {rank} - Completed {name}/{subname} ({idx}/{total_cases})", flush=True)
                    else:
                        print(f"Rank: {rank} - Skipped {name}/{subname} ({idx}/{total_cases}) - No valid data", flush=True)

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
    #stress = ase_object.calc.results["stress"]
    fermi_energy = ase_object.calc.eFermi
    free_energy = ase_object.calc.results["free_energy"]
    energy = ase_object.calc.results["energy"]
    node_feature_matrix = np.concatenate(
        (proton_numbers, forces), axis=1
    )
    data_object.x = tensor(node_feature_matrix).float()

    # Extract directory path from OUTCAR filepath
    dirpath = os.path.dirname(filepath)
    rmsd_file_path = os.path.join(dirpath, 'root_mean_squared_displacement.txt')
    
    mean_squared_displacement_file = open(rmsd_file_path, 'r')
    Lines = mean_squared_displacement_file.readlines()

    # Strips the newline character
    for line in Lines:
        data_object.y = tensor([float(line.strip())])

    return data_object


if __name__ == '__main__':
    elements_list = ['Nb', 'Zr']
    source_path = './bcc_enthalpy_' + elements_list[0] + elements_list[1]
    plot_data(source_path, elements_list)
