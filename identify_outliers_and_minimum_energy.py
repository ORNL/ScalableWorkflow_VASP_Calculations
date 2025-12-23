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

import math
import pandas as pd

def compute_standard_deviation(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def filter_outliers_and_find_min(data, number_standard_deviations):
    # Extract the second elements (float values) from the tuples
    values = [value for _, value in data]

    # Compute the mean and standard deviation
    mean = sum(values) / len(values)
    std_dev = compute_standard_deviation(values)

    # Determine the range for non-outliers
    lower_bound = mean - number_standard_deviations * std_dev
    upper_bound = mean + number_standard_deviations * std_dev

    # Filter out the outliers
    filtered_data = [item for item in data if lower_bound <= item[1] <= upper_bound]

    # Find the element with the minimum value after removing outliers
    min_element = min(filtered_data, key=lambda x: x[1])

    # Return the outliers and the element with the minimum value
    outliers = [item for item in data if item not in filtered_data]

    return outliers, min_element


def save_to_csv(data, filename):
    # Create a pandas DataFrame from the list of tuples
    df = pd.DataFrame(data, columns=["Case Name", "formation energy"])

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


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

def collect_statistics_per_composition(source_path, elements_list, number_standard_deviations):

    assert len(elements_list) == 2, "Plots are supported only for binaries"

    outliers, minima = screen_data(source_path, number_standard_deviations)

    return outliers, minima

def screen_data(raw_data_path, number_standard_deviations):
    """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
    After that the serialized data is stored to the serialized_dataset directory.
    """

    outliers = []
    minima = []

    dirs = None
    if rank == 0:
        dirs = [f.name for f in os.scandir(raw_data_path) if f.is_dir()]

    dirs = comm.bcast(dirs, root=0)

    rx = list(nsplit(range(len(dirs)), size))[rank]

    for name in sorted(dirs)[rx.start:rx.stop]:
        print("f Rank: ", rank, " - name: ", name, flush=True)
        if name == ".DS_Store":
            continue
        # if the directory contains subdirectories, explore their content
        if os.path.isdir(os.path.join(raw_data_path, name)):
            if name == ".DS_Store":
                continue
            dir_name = os.path.join(raw_data_path, name)
            for subname in os.listdir(dir_name):
                if subname == ".DS_Store":
                    continue
                subdir_name = os.path.join(dir_name, subname)
                structures_log = []
                for subsubname in os.listdir(subdir_name):
                    # Here we span all the atomistic configurations per chemical composition
                    if subsubname == ".DS_Store":
                        continue
                    subsubdir_name = os.path.join(subdir_name, subsubname)
                    for subsubsubname in os.listdir(subsubdir_name):
                        if os.path.isfile(os.path.join(subsubdir_name, subsubsubname)):
                            data_object = transform_input_to_data_object_base(
                                filepath=os.path.join(subsubdir_name, subsubsubname)
                            )
                            if not isinstance(data_object, type(None)):
                                structures_log.append((os.path.join(subsubdir_name, subsubsubname), data_object.y.item()))

                outliers_per_composition, min_element_per_composition = filter_outliers_and_find_min(structures_log, number_standard_deviations)
                outliers.extend(outliers_per_composition)
                minima.append(min_element_per_composition)

    return outliers, minima


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
    elements_list = ['Ta', 'V']
    # source_path = './bcc_enthalpy/'+'-'.join(elements_list)
    source_path = './10.13139_OLCF_2222910/bcc_' + '-'.join(elements_list)

    for i in range(1,5):
        number_standard_deviations = i
        outliers, minima = collect_statistics_per_composition(source_path, elements_list, number_standard_deviations)

        # Save the outliers to a CSV file
        save_to_csv(outliers, "outliers_" + ''.join(elements_list) + f"_{number_standard_deviations}_sigmas" + ".csv")

        # Save the minimum element to a CSV file
        save_to_csv(minima, "minima_" + ''.join(elements_list) + f"_{number_standard_deviations}_sigmas" + ".csv")

    print("Outliers and minimum element have been saved to CSV files.")
