# FEniCSx solver using mixed function spaces
# solves for a 2d space with Qij being a 2x2 symmetric and traceless tensor

import numpy as np
import matplotlib.pyplot as plt
import ufl,time,meshio
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc
from tqdm.auto import tqdm

axispts = 11
T = 1
nsteps = 100
dt = 1e-6
T = nsteps*dt
isave = True
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
S0 = 0.53 # order parameter

def xdmf_eig(filename,nsteps):
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        it = 0
        for k in tqdm(range(reader.num_steps)):
            t, point_data, cell_data = reader.read_data(k)
            data = point_data['f']
            eig_data = np.zeros((data.shape[0],2))
            for p in range(data.shape[0]):
                Q = np.reshape(data[p,:],(2,2))
                w,v = np.linalg.eig(Q) #w gives eigenvalues, v gives eigenvectors (v[:,i])
                eig_data[p,:] = v[:,np.argmax(w)]

            new_mesh = mesh = meshio.Mesh(points,cells,point_data={"N": eig_data})
            vtk_filename = "qtensor"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            mesh.write(vtk_filename)
            it += 1


def main():
    # define a mesh
    msh = mesh.create_unit_square(comm = MPI.COMM_WORLD,
                                  nx = axispts,
                                  ny = axispts)
    
    P = ufl.FiniteElement('Lagrange', msh.ufl_cell(), 2)
    FS = fem.FunctionSpace(msh,ufl.MixedElement([P,P])) #CG == Lagrange

    #Q0FS = fem.FunctionSpace(msh,P) #CG == Lagrange
    #Q1FS = fem.FunctionSpace(msh,P) #CG == Lagrange

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

    def Qxy(x):
        #n = np.zeros(msh.geometry.dim) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x[0].shape))
        azi_angle = np.random.uniform(0,2*np.pi,x[0].shape)
        nx = np.sin(polar_angle)*np.cos(azi_angle)
        ny = np.sin(polar_angle)*np.sin(azi_angle)
        Qxy = S0*(nx*ny)
        return Qxy

    Q_new = fem.Function(FS) # current time-step result
    Q_old = fem.Function(FS) # previous time-step result
    (q0_new,q1_new) = ufl.split(Q_new)
    (q0_old,q1_old) = ufl.split(Q_old)
    (v0,v1) = ufl.TestFunctions(FS)
    #initializing Q for random director
    for i in range(Q_new.ufl_shape[0]):
        Q_new.sub(i).x.array[:] = 0.0
        Q_old.sub(i).x.array[:] = 0.0
    
    # distributing initial condition
    Q_new.sub(0).interpolate(Qxx)
    Q_new.sub(0).x.scatter_forward()
    Q_new.sub(1).interpolate(Qxy)
    Q_new.sub(1).x.scatter_forward()

    Q = ufl.as_tensor(((q0_new, q1_new),
                      (q1_new, -q0_new)))
    Q_n = ufl.as_tensor(((q0_old, q1_old),
                      (q1_old, -q0_old)))
    V = ufl.as_tensor(((v0, v1),
                      (v1, -v0)))

    # writing initial conditions to file
    if isave:
        FS_out = fem.FunctionSpace(msh, P)
        q0_out = fem.Function(FS_out)
        q1_out = fem.Function(FS_out)
        xdmf_q0_file = io.XDMFFile(msh.comm, "qtensor0.xdmf",'w')
        xdmf_q1_file = io.XDMFFile(msh.comm, "qtensor1.xdmf",'w')
        xdmf_q0_file.write_mesh(msh)
        xdmf_q1_file.write_mesh(msh)
        print(type(Q_new.sub(0)))
        print(type(Q_new.sub(1).x.array[:]))
        print(Q_new.sub(0).x.array[:].shape)
        q0_out.interpolate(Q_new.sub(1).x.array[:])
        xdmf_q0_file.write_function(Q_new.sub(0),0.0)
        xdmf_q1_file.write_function(Q_new.sub(1),0.0)
        #xdmf_q0_file.close()


    # defining some constants
    A = fem.Constant(msh,PETSc.ScalarType(1.0))
    B = fem.Constant(msh, PETSc.ScalarType(1.0))
    C = fem.Constant(msh, PETSc.ScalarType(1.0))
    L = fem.Constant(msh, PETSc.ScalarType(1.0))
    k = fem.Constant(msh, PETSc.ScalarType(dt))

    # backwards euler part of residuals
    F1 = ufl.inner((Q - Q_n)/k,V)*ufl.dx 
    # bulk free energy part
    F2 = -1*ufl.inner((A*Q + B*ufl.dot(Q,Q) + C*(ufl.inner(Q,Q)*Q)),V)*ufl.dx
    # distortion/elastic term
    F3 = -1*(ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
    # construct the residual
    F = F1+F2+F3
    #print(fem.assemble_scalar(F2+F3))
    
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
        print(f"Step {int(t/dt)}: num iterations: {r[0]}")
        if (isave):
            xdmf_q0_file.write_function(Q_new.sub(0),t)
            xdmf_q1_file.write_function(Q_new.sub(1),t)
        Q_old.x.array[:] = Q_new.x.array

    xdmf_q0_file.close()
    xdmf_q1_file.close()
    print("Done!")
    
if __name__ == '__main__':
    main()