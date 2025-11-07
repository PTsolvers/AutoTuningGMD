# AutoTuningGMD

This repository archives scripts that allow to reproduce results of the submitted manuscript to Geoscientific Model Development.
Automatic Dynamic Relaxation (DR) and Powell-Hestenes / Dynamic Relaxation (PHDR) solvers using Finite Difference (FD) and Face-Centered Finite Volume (FCFV) discretisations.

### 1D 
FD and FCFV Poisson solvers with heterogeneous coefficients usin the DR method.
![](./img/BasicDyRel.png)

### 2D
FD models nased on th PHDR and DR methods: compressible/incompressible deformation, frictional plasticity, two-phase flow 

### 3D 
FD and FCFV models of incompressible Stokes flow with numerous inclusions of variable viscosity.

## Quickstart
1. Clone/Download the repository
2. Open the folder in a dedicated VScode window 
3. Open Julia's REPL and type to switch to package mode: `]`
5. Install all necessary dependencies, type: `instantiate`