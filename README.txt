Description
-----------

These are the scripts to obtain forward solution for the four sphere head
models to obtain the EEG potentials from the point dipole sources located
inside the brain of 4 different compositions (brain, csf, skull, scalp).

We implemented, the correct Analytical formulation for this, and an equivalent
numerical Finite element model, and provide with the previously presented
analytical methods. We validate our method Analytical and the numerical with
each other, and compare it with the previous ones. We also test the limiting
case when all the four spheres have the same conductivity.

Our presented results can be used for placing a dipole in any orientation, and
the electrodes at place (for analytical the electrodes must be > radial distance
of the dipole). These can be modified in the parameters.py file. While the
analytical solution is recommended for its fast computational time, the
numerical solution serves as a good starting point for further improvements for
a more sophisticated head models and more complicated conductivity profiles.

Authors
-------
Written by Chaitanya Chintaluri(1), Solveig Næss(2) and Torbjørn Ness(3)
Minor updates by Jakub M. Dzik(1).

1. Nencki Institute of Experimental Biology, Warsaw, Poland.
2. University of Oslo, Oslo, Norway.
3. Norwegian University of Life Sciences, Ås, Norway.

License
-------
Copyright 2017 Chaitanya Chintaluri, Solveig Næss, and Torbjørn Ness

This software is provided under the GNU General Purpose License version 3.0,
You will find a copy of this license within this folder, or from online here: 
https://www.gnu.org/licenses/gpl-3.0.txt


Requirements
------------

To run the scripts pull the Docker image:
$ docker pull quay.io/fenicsproject/stable

You will also need:
- Python 3.6
- matplotlib python package
- gmsh 2.13.1
- dolfin 2017.2


Files and work flow
-------------------

1) Generate the mesh (USE gmsh 2.13.1 to create a .msh file from .geo, and dolfin
to create .xml files from .msh)

$ cd mesh
$ sh mesh_it.sh
***NOW YOU WAIT***


1a) Run the docker image

$ cd ..
$ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
fenics@...$ cd /home/fenics/shared/


2) Finite Element Method

fenics@...$ python3 numerical_fem.py

***NOW YOU WAIT***
(uses params from parameters.py)
(gets you ./results/Numerical_*.npz ; * is rad, tan, mix)

Computes the potentials on the scalp hemisphere for the cases of 
skull sigma = scalp sigma / 20; "fem_20"
skull sigma = scalp sigma / 40; "fem_40"
skull sigma = scalp sigma / 80; "fem_80"

For the radial, tangential and the 45 deg. oriented dipole

3) Analytical methods from Nunez and Srinivasan '06, Appendix G

fenics@...$ python3 analytical_NunSri06.py

(uses params from parameters.py)
(gets you ./results/Analytical_NunSri06_rad.npz)

Computes the potentials on the scalp hemisphere for the cases of 
skull sigma = scalp sigma / 20; "phi_20"
skull sigma = scalp sigma / 40; "phi_40"
skull sigma = scalp sigma / 80; "phi_80"
All sigmas equal, case; "phi_lim"

For the radial dipole ONLY.

4) Analytical methods from Srinivasan '98, Appendix

fenics@...$ python3 analytical_Sri98.py

(uses params from parameters.py)
(gets you ./results/Analytical_Sri98_rad.npz)

Computes the potentials on the scalp hemisphere for the cases of 
skull sigma = scalp sigma / 20; "phi_20"
skull sigma = scalp sigma / 40; "phi_40"
skull sigma = scalp sigma / 80; "phi_80"
All sigmas equal, case; "phi_lim"

For the radial dipole ONLY.

5) Analytical methods, Corrected solution

fenics@...$ python3 analytical_correct.py

!!!!***!!!!
(uses params from parameters.py)
(gets you ./results/Analytical_*.npz; *  is rad, tan, mix)

Computes the potentials on the scalp hemisphere for the cases of 
skull sigma = scalp sigma / 20; "fem_20"
skull sigma = scalp sigma / 40; "fem_40"
skull sigma = scalp sigma / 80; "fem_80"

For the radial, tangential and the 45 deg. oriented dipole

5) Make plots! 

(i)

fenics@...$ python3 figure_solveig_unit_upscaled.py

Includes the scaling used in the manuscript

fenics@...$ python3 figure2.py

!!!!***!!!!
Compares that the analytical solution and the FEM solution have converged

!!!!***!!!!
An alternate view of this plot available see,

fenics@...$ python3 figure2_alt.py

(ii)

fenics@...$ python3 figure3.py --sri98-no-bn1

Compares the previous methods to the proposed solution.

(iii)

fenics@...$ python3 figure4.py --sri98-no-bn1

Compares the limiting case of all equal conductivity for the analytical
methods.

(iv)

fenics@...$ python3 plot_test.py

Compares the two analytical solution implementation for the various
configurations of the dipole orientations.

Warning: plot_test.py uses data files provided in the repository which are
no longer generated with the aforementioned scripts.


NOTES
-----

The mesh used ./mesh/sphere_4.geo actually has 5 spheres instead of 4.
The inner most sphere is there for the sake of simplifying the mesh size and
the convergence time for the FEM simulation.

A lower resolution of this mesh is also available ./mesh/sphere_4_lowres.geo
which establishes the convergence of the solution.

In order to use it, run:

$ cd mesh
$ sh mesh_it.sh sphere_4_lowres
$ cd ..
$ docker run -ti --env HOST_UID=$(id -u) --env HOST_GID=$(id -g) -v $(pwd):/home/fenics/shared:Z quay.io/fenicsproject/stable
fenics@...$ cd /home/fenics/shared/
fenics@...$ python3 numerical_fem.py mesh/sphere_4_lowres.xml -d results_lowres
fenics@...$ python3 analytical_NunSri06.py -d results_lowres
fenics@...$ python3 analytical_Sri98.py -d results_lowres
fenics@...$ python3 analytical_correct.py -d results_lowres
fenics@...$ python3 figure_solveig_unit_upscaled.py -d results_lowres
fenics@...$ python3 figure2.py -d results_lowres
fenics@...$ python3 figure2_alt.py -d results_lowres
fenics@...$ python3 figure3.py -d results_lowres --sri98-no-bn1
fenics@...$ python3 figure4.py -d results_lowres --sri98-no-bn1