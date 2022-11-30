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


def nPr(n, r):
    return int(factorial(n) / factorial(n - r))


def nCr(n, r):
    return int(factorial(n) / (factorial(n - r) * factorial(r)))


def create_vasp_case(case_path, composition_ase_object):
    os.makedirs(case_path, exist_ok=False)
    shutil.copy(source_path + "/INCAR", case_path)
    shutil.copy(source_path + "/POTCAR", case_path)
    shutil.copy(source_path + "/KPOINTS", case_path)
    write_vasp(case_path + "/POSCAR", direct=True, atoms=composition_ase_object)


def generate_VASP_randomized_binary_configurations(source_path, destination_path, atomic_increment, symmetry_savings,
                                                   num_atomic_configurations=1):
    " Read atomic configuration from POSCAR file "

    os.makedirs(destination_path, exist_ok=False)

    assert os.path.exists(source_path + "/INCAR")
    assert os.path.exists(source_path + "/POTCAR")
    assert os.path.exists(source_path + "/KPOINTS")
    assert os.path.exists(source_path + "/POSCAR")

    ase_object = read_vasp(source_path + "/" + "POSCAR")

    # Extract set of atom elements
    elements_list = list(set(list(ase_object.numbers)))
    assert 2 == len(elements_list), "This function can only be used for binary alloys"
    num_atoms = ase_object.numbers.shape[0]

    chemical_composition = np.zeros_like(ase_object.numbers)

    for num_element_atoms in range(0, num_atoms + 1, atomic_increment):

        for composition_idx in range(0, num_element_atoms):
            chemical_composition[composition_idx] = elements_list[0]

        for composition_idx in range(num_element_atoms, num_atoms):
            chemical_composition[composition_idx] = elements_list[1]

        composition_ase_object = ase_object
        composition_ase_object.numbers = chemical_composition
        chemical_composition_dir = destination_path + "/" + str(composition_ase_object.symbols)

        os.makedirs(chemical_composition_dir, exist_ok=False)

        if nCr(num_atoms, num_element_atoms) > num_atomic_configurations:
            for randomized_configuration_id in range(0, num_atomic_configurations):
                case_path = chemical_composition_dir + "/case-" + str(randomized_configuration_id + 1)
                # When the histogram of the distribution is cut,
                # a check is performed to see if the total number of configurations is lower or higher than the number of atomic configurations needed
                np.random.shuffle(composition_ase_object.positions)
                create_vasp_case(case_path, composition_ase_object)
        elif symmetry_savings:
            # FIXME not implemented yet
            raise TypeError("Error: symmetry savings not implemented yet")
        elif not symmetry_savings:
            indices_list = [index for index in range(0, num_atoms)]
            if num_element_atoms < int(num_atoms / 2):
                element_original_indices = np.where(chemical_composition == elements_list[0])[0]
                num_swapping = num_element_atoms
            else:
                element_original_indices = np.where(chemical_composition == elements_list[1])[0]
                num_swapping = num_atoms - num_element_atoms
            permutation_count = 0
            # if the total number of permutations is smaller than the total number of configurations needed,
            # randomization is not needed because we need to include all the permutations in the dataset
            for permutation in permutations(indices_list, num_swapping):
                case_path = chemical_composition_dir + "/case-" + str(permutation_count + 1)
                new_ase_object = composition_ase_object
                # we need to exchange the rows of the position matrix associated with the atom types that need to be swapped
                for original_index, new_index in zip(element_original_indices, list(permutation)):
                    new_ase_object.positions[[original_index, new_index]] = composition_ase_object.positions[
                        [new_index, original_index]]
                create_vasp_case(case_path, new_ase_object)
                permutation_count += 1
            assert nCr(num_atoms, num_element_atoms) == permutation_count


if __name__ == '__main__':
    current_directory = os.getcwd()
    source_path = current_directory + '/case-prototype'
    destination_path = current_directory + "/single_phase_dataset"
    atomic_increment = 1
    num_atomic_configurations_per_composition = 150
    symmetry_savings = False
    generate_VASP_randomized_binary_configurations(source_path, destination_path, atomic_increment, symmetry_savings,
                                                   num_atomic_configurations_per_composition)
