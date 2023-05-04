# FEniCSx solver using TensorFunctionSpaces

import numpy as np
import matplotlib.pyplot as plt
import ufl
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc

axispts = 11
T = 10
nsteps = 500
dt = T/nsteps
isave = False
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
S0 = 0.53 # order parameter

def main():
    # define a mesh
    msh = mesh.create_unit_cube(comm = MPI.COMM_WORLD,
                                  nx = axispts,
                                  ny = axispts,
                                  nz = axispts)
    
    # define a Tensor function space
    #FS = fem.TensorFunctionSpace(msh,"Lagrange", 2, shape=(3,3), symmetry=True)
    P = ufl.FiniteElement('Lagrange', msh.ufl_cell(), 2)
    FS = fem.FunctionSpace(msh,ufl.MixedElement([P,P,P,P,P])) #CG == Lagrange
    print("T:",T)
    print("nsteps:",nsteps)
    print("dt",dt)
    #class InitialConditions(UserExpression):
        
    def Qxx(x):
        polar_angle = np.arccos(np.random.uniform(-1,1,x[0].shape))
        azi_angle = np.random.uniform(0,2*np.pi,x[0].shape)
        nx = np.sin(polar_angle)*np.cos(azi_angle)
        Qxx = S0*(nx*nx - 1/3)
        return Qxx

    def Qyy(x):
        #n = np.zeros(msh.geometry.dim) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x[0].shape))
        azi_angle = np.random.uniform(0,2*np.pi,x[0].shape)
        ny = np.sin(polar_angle)*np.sin(azi_angle)
        Qyy = S0*(ny*ny - 1/3)
        return Qyy
    
    def Qzz(x):
        polar_angle = np.arccos(np.random.uniform(-1,1,x[0].shape))
        azi_angle = np.random.uniform(0,2*np.pi,x[0].shape)
        ny = np.sin(polar_angle)*np.sin(azi_angle)
        Qyy = S0*(ny*ny - 1/3)
        
    def Qxy(x):
        pass
    Q_new = fem.Function(FS) # current time-step result
    Q_old = fem.Function(FS) # previous time-step result
    (q0_new,q1_new,q2_new,q3_new,q4_new) = ufl.split(Q_new)
    (q0_old,q1_old,q2_old,q3_old,q4_old) = ufl.split(Q_old)
    (v0,v1,v2,v3,v4) = ufl.TestFunctions(FS)
    #initializing Q for random director
    #only need to fill q0_new, q1_new as the rest will be 0/populated with q0/q1
    for i in range(Q_new.ufl_shape[0]):
        Q_new.sub(i).x.array[:] = 0.0
        Q_old.sub(i).x.array[:] = 0.0
    
    # distributing initial condition
    Q_new.sub(0).interpolate(Qxx)
    Q_new.sub(0).x.scatter_forward()
    Q_new.sub(1).interpolate(Qyy)
    Q_new.sub(1).x.scatter_forward()

    Q = ufl.as_tensor(((q0_new, q4_new, q3_new),
                       (q2_new, q1_new, q2_new),
                       (q3_new, q4_new, -q0_new-q1_new)))
    Q_n = ufl.as_tensor(((q0_old, q4_old, q3_old),
                         (q2_old, q1_old, q2_old),
                         (q3_old, q4_old, -q0_old-q1_old)))
    V = ufl.as_tensor(((v0, v4, v3),
                       (v2, v1, v2),
                       (v3, v4, -v0-v1)))

    # writing initial conditions to file
    #xdmf_Q_file = io.XDMFFile(msh.comm, "qtensor.xdmf",'w')
    #xdmf_Q_file.write_mesh(msh)
    #xdmf_Q_file.write_function(Q_new)
    #xdmf_Q_file.close()

    # defining some constants
    A = fem.Constant(msh,PETSc.ScalarType(1.0))
    B = fem.Constant(msh, PETSc.ScalarType(1.0))
    C = fem.Constant(msh, PETSc.ScalarType(1.0))
    L = fem.Constant(msh, PETSc.ScalarType(1.0))
    k = fem.Constant(msh, PETSc.ScalarType(dt))

    # backwards euler part of residual
    F1 = ufl.inner((Q - Q_n)/k,V)*ufl.dx 
    # bulk free energy part
    F2 = -1*ufl.inner((A*Q + B*ufl.dot(Q,Q) + C*(ufl.inner(Q,Q)*Q)),V)*ufl.dx
    # distortion/elastic term
    F3 = -1*(ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
    # construct the residual
    F = F1+F2+F3
    #print(fem.assemble_scalar(F2+F3))
    #exit()
    # Create nonlinear problem and Newton solver
    problem = fem.petsc.NonlinearProblem(F, Q_new)
    solver = nls.petsc.NewtonSolver(msh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-6

    # We can customize the linear solver used inside the NewtonSolver by
    # modifying the PETSc options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    ksp.setFromOptions()

    #step in time
    print("starting")
    t = 0.0
    Q_old.x.array[:] = Q_new.x.array[:]
    while (t < T):
        t += dt
        r = solver.solve(Q_new)
        E = assemble(F2+F3)
        print(f"Step {int(t/dt)}: num iterations: {r[0]}")
        Q_old.x.array[:] = Q_new.x.array

    print("Done!")
    
if __name__ == '__main__':
    main()