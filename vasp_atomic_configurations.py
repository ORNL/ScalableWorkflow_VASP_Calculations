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

from ase.io.vasp import read_vasp, write_vasp

plt.rcParams.update({"font.size": 20})


def generate_VASP_randomized_binary_configurations(source_path, destination_path, atomic_increment, num_atomic_configurations=1):

    " Read atomic configuration from POSCAR file "

    os.makedirs(destination_path, exist_ok=False)

    assert os.path.exists(source_path + "/INCAR")
    assert os.path.exists(source_path + "/POTCAR")
    assert os.path.exists(source_path + "/KPOINTS")
    assert os.path.exists(source_path + "/POSCAR")


    ase_object = read_vasp(source_path+"/"+"POSCAR")

    # Extract set of atom elements
    elements_list = list(set(list(ase_object.numbers)))
    assert 2 == len(elements_list), "This function can only be used for binary alloys"
    num_atoms = ase_object.numbers.shape[0]

    chemical_composition = np.zeros_like(ase_object.numbers)

    for num_element_atoms in range (0, num_atoms+1, atomic_increment):

        for composition_idx in range(0,num_element_atoms):
            chemical_composition[composition_idx] = elements_list[0]

        for composition_idx in range(num_element_atoms, num_atoms):
            chemical_composition[composition_idx] = elements_list[1]

        composition_ase_object = ase_object
        composition_ase_object.numbers = chemical_composition
        chemical_composition_dir = destination_path+"/"+str(composition_ase_object.symbols)

        os.makedirs(chemical_composition_dir, exist_ok=False)

        for randomized_configuration_id in range(0, num_atomic_configurations):
            case_path = chemical_composition_dir + "/case-" + str(randomized_configuration_id+1)
            os.makedirs(case_path, exist_ok=False)
            np.random.shuffle(composition_ase_object.positions)
            shutil.copy(source_path+"/INCAR", case_path)
            shutil.copy(source_path + "/POTCAR", case_path)
            shutil.copy(source_path + "/KPOINTS", case_path)
            write_vasp(case_path+"/POSCAR", direct=True, atoms=composition_ase_object)


if __name__ == '__main__':
    source_path = 'case-prototype'
    destination_path = "./single_phase_dataset"
    atomic_increment = 4
    num_atomic_configurations_per_composition = 10
    generate_VASP_randomized_binary_configurations(source_path, destination_path, atomic_increment, num_atomic_configurations_per_composition)
