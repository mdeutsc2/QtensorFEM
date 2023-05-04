# qtensor-fem
## Finite Element Method of Q-tensor Dynamics in Nematic Liquid Crystals

### Outline
* ```/data``` contains data from simulations
* ```/julia``` contains example from-scratch FEM simulation in Julia
* ```/scripts``` contains Jupyter Notebooks and utility scripts
* ```/src``` contains python code to implement solvers using FEniCSx
    - ```fenics_qij.py``` uses a MixedElement space of Scalar elements to solve for elements of Q-tensor independently
    - ```fenics_qij2.py``` uses a TensorElement space to simplify number of solvers and residuals
    - ```hlbm_qij.py``` and ```qij.py``` contain from-scratch finite difference method solvers

### TODO
* module for each solver
* make forms independent of solver so the each solver can be written indepdently
* functions that return forms
* pull forms into classes the do the solving
* small scripts for each problem


