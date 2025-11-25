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

from ase.io.vasp import read_vasp, write_vasp

from itertools import combinations, permutations

plt.rcParams.update({"font.size": 20})

from math import factorial

# Fixing the seed
np.random.seed(42)

def nPr(n, r):
    return int(factorial(n) / factorial(n - r))


def nCr(n, r):
    return int(factorial(n) / (factorial(n - r) * factorial(r)))


def create_vasp_case(case_path, composition_ase_object):
    os.makedirs(case_path, exist_ok=False)
    shutil.copy(source_path + "/START", case_path)
    shutil.copy(source_path + "/POTCAR", case_path)
    shutil.copy(source_path + "/KPOINTS", case_path)
    write_vasp(case_path + "/POSCAR", direct=True, sort=True, atoms=composition_ase_object)


def generate_VASP_randomized_ternary_configurations(source_path, destination_path, atomic_increment, symmetry_savings,
                                                   num_atomic_configurations=1):
    " Read atomic configuration from POSCAR file "

    os.makedirs(destination_path, exist_ok=False)

    assert os.path.exists(source_path + "/START")
    assert os.path.exists(source_path + "/POTCAR")
    assert os.path.exists(source_path + "/KPOINTS")
    assert os.path.exists(source_path + "/POSCAR")

    ase_object = read_vasp(source_path + "/" + "POSCAR")

    # Extract set of atom elements
    elements_list = list(set(list(ase_object.numbers)))
    assert 3 == len(elements_list), "This function can only be used for ternary alloys"
    num_atoms = ase_object.numbers.shape[0]

    chemical_composition = np.zeros_like(ase_object.numbers)

    for num_first_element_atoms in range(0, num_atoms + 1, atomic_increment):

        for num_second_element_atoms in range(0, num_atoms - num_first_element_atoms, atomic_increment):

            for idx in range(0, num_first_element_atoms):
                chemical_composition[idx] = elements_list[0]

            for idx in range(num_first_element_atoms, num_first_element_atoms + num_second_element_atoms):
                chemical_composition[idx] = elements_list[1]

            for idx in range(num_first_element_atoms + num_second_element_atoms, num_atoms):
                chemical_composition[idx] = elements_list[2]

            composition_ase_object = ase_object
            composition_ase_object.numbers = chemical_composition
            chemical_composition_dir = destination_path + "/" + str(composition_ase_object.symbols)

            os.makedirs(chemical_composition_dir, exist_ok=True)

            if nCr(num_atoms, num_first_element_atoms) * nCr(num_atoms - num_first_element_atoms, num_second_element_atoms) > num_atomic_configurations:
                for randomized_configuration_id in range(0, num_atomic_configurations):
                    case_path = chemical_composition_dir + "/case-" + str(randomized_configuration_id + 1)
                    # When the histogram of the distribution is cut,
                    # a check is performed to see if the total number of configurations is lower or higher than the number of atomic configurations needed
                    np.random.shuffle(composition_ase_object.positions)
                    create_vasp_case(case_path, composition_ase_object)
            else:
                pass


if __name__ == '__main__':
    current_directory = os.getcwd()
    source_path = current_directory + '/template_Nb-Ta-Zr'
    destination_path = current_directory + '/bcc/Nb-Ta-Zr'
    atomic_increment = 8
    num_atomic_configurations_per_composition = 100
    symmetry_savings = False
    generate_VASP_randomized_ternary_configurations(source_path, destination_path, atomic_increment, symmetry_savings,
                                                   num_atomic_configurations_per_composition)
