## ScalableWorkflow_VASP_Calculations

## Description
This repository provides tools to automate the generation of inputs to run ensemble first-principles calculations using the Vienna Ab-Initio Simulation Pakcage (VASP) for disordered atomic configurations. 

## Installation requirements
The requirements are specified in the **requirements.txt** file.

## Usage
The code to prepare the inputs of the VASP calculations and performing post-processing statistical analysis of the dataset is open-source and available at the ORNL-GitHub repository 
[ORNL-GitHub repository](https://github.com/ORNL/ScalableWorkflow_VASP_Calculations)
The code contains the following Python script:
- **generate_vasp_atomic_configurations_binary.py**: generates the hierarchy of directories (with one directory per chemical composition and multiple subdirectories names **case-N**, with **N** between 1 and 100, for each chemical composition of the binary alloiy)
- **generate_vasp_atomic_configurations_ternary.py**: generates the hierarchy of directories (with one directory per chemical composition and multiple subdirectories names **case-N**, with **N** between 3 and 100, for each chemical composition of the ternary alloy)
- **compute_formation_energy.py**: computes the formation energy for each fully optimized atomic structure by subtracting the linear term of the total energy
- **plot_formation_energy_binary.py**: plots the histogram for the distribution of values of the formation energy for the binary alloys and generates the scatterplot of the formation energies for each atomic structure as a function of the chemical concentration of one of the two constituents of the binary alloys
- **plot2D_sliced_formation_energy_ternary_ase_objects.py**:
plots the histogram for the distribution of values of the formation energy for the binary alloys and generates the scatterplots of the formation energies for each atomic structure by fixing the chemical concentration of one constituent, and plotting the formation energy values as a function of the chemical concentration of another constituent of the ternary alloys

## Support
For questions on how to use the tools, please reach out to [Lupo Pasini, Massimiliano](mailto:lupopasinim@ornl.gov)

## Contributing
We are open to contributions. Please feel free to submit pull requests (PRs) or email us. 

## Authors
- Lupo Pasini, Massimiliano

## Acknowledgement
This research is sponsored by the Artificial Intelligence Initiative as part of the Laboratory Directed Research and Development (LDRD) Program of Oak Ridge National Laboratory, managed by UT-Battelle, LLC, for the US Department of Energy under contract DE-AC05-00OR22725.
This work used resources of the Oak Ridge Leadership Computing Facility, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725, under Directorate Discretionary awards MAT025 (Materials Science) and LRN026 (Machine Learning), and INCITE award MAT201. This work also used resources of the National Energy Research
Scientific Computing Center, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231, under award ERCAP0025216.

## License
For open source projects, say how it is licensed.

