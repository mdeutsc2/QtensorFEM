# FEniCSx solver using TensorFunctionSpaces
# in 2D

import numpy as np
import ufl, time, meshio
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc
from tqdm.auto import tqdm

axispts = 21
T = 10
nsteps = 200
dt = 1e-6#T/nsteps
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
            eig_data = np.zeros((data.shape[0],3))
            for p in range(data.shape[0]):
                Q = np.reshape(data[p,:],(3,3))
                w,v = np.linalg.eig(Q) #w gives eigenvalues, v gives eigenvectors (v[:,i])
                eig_data[p,:] = v[:,np.argmax(w)]

            new_mesh = meshio.Mesh(points,cells,point_data={"N": eig_data})
            vtk_filename = "qtensor"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            new_mesh.write(vtk_filename)
            it += 1

def main():
    # Sub domain for Periodic boundary condition
    # class PeriodicBoundary(SubDomain):

    #     # Left boundary is "target domain"
    #     def inside(self, x, on_boundary):
    #         return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    #     # Map right boundary to left boundary
    #     def map(self, x, y):
    #         y[0] = x[0] - 1.0
    #         y[1] = x[1]
    # # define a mesh
    # # msh = mesh.create_unit_square(comm = MPI.COMM_WORLD,
    # #                               nx = axispts,
    # #                               ny = axispts)
    # pbc = PeriodicBoundary()
    msh = mesh.create_unit_square(comm = MPI.COMM_WORLD,
                                  nx = axispts,
                                  ny = axispts)
    x = ufl.SpatialCoordinate(msh) 
    #P = ufl.TensorElement('CG', ufl.tetrahedron, 1, symmetry=True)
    P = ufl.TensorElement('CG',msh.ufl_cell(),1,symmetry=True)
    FS = fem.FunctionSpace(msh,P) #CG == Lagrange
    
    print("T:",T)
    print("nsteps:",nsteps)
    print("dt",dt)

    # initial conditions
    def initQ3d(x):
        # values[0] = tensor[0,0]  --> 0 1 2
        # values[1] = tensor[0,1]      3 4 5
        # values[2] = tensor[0,2]      6 7 8
        # values[3] = tensor[1,0] ...
        values = np.zeros((3*3,
                      x.shape[1]), dtype=np.float64)
        n = np.zeros((3,x[0].shape[0])) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x[0].shape))
        azi_angle = np.random.uniform(0,2*np.pi)
        n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        n[2,:] = np.cos(polar_angle)
        #n = np.linalg.norm(n)
        #Qxx = S0*(n[0]*n[0] - 1/3)
        values[0] = S0*(n[0,:]*n[0,:]-1/3)
        values[1] = S0*(n[0,:]*n[1,:])
        values[2] = S0*(n[0,:]*n[2,:])
        values[3] = S0*(n[1,:]*n[0,:])
        values[4] = S0*(n[1,:]*n[1,:]-1/3)
        values[5] = values[3]
        values[6] = values[2]
        values[7] = values[1]
        values[8] = -values[0]-values[4]
        return values

    def initQ2d(x):
        values = np.zeros((2*2,x.shape[1]),dtype=np.float64)
        n = np.zeros((2,x[0].shape[0]))
        polar_angle = np.arccos(np.random.uniform(-1,1,x[0].shape))
        azi_angle = np.random.uniform(0,2*np.pi)
        n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        values[0] = S0*(n[0,:]*n[0,:]-1/3)
        values[1] = S0*(n[0,:]*n[1,:])
        values[2] = S0*(n[0,:]*n[0,:]-1/3)
        values[3] = S0*(n[0,:]*n[1,:])
        return values

    Q = fem.Function(FS) # current time-step result
    Q_n = fem.Function(FS) # previous time-step result
    V = ufl.TestFunction(FS) # test function to weight calcuations through the lattice

    #initializing Q for random director and distributing initial condition
    Q.interpolate(initQ2d)
    #Q.x.scatter_forward()

    print("len(x.array[:] ", len(Q.x.array[:]))
    print(type(Q.vector))
    print(Q.vector.size)
    print("DOF coords: ",FS.tabulate_dof_coordinates().shape)
    print("Global size: ",FS.dofmap.index_map.size_global)
    print("Local size: ", FS.dofmap.index_map.size_local)
    print("Local range: ", FS.dofmap.index_map.local_range)
    print(Q.vector.getOwnershipRange())
    print(0-Q.vector.getOwnershipRange()[0])


    # writing initial conditions to file
    if (isave):
        xdmf_Q_file = io.XDMFFile(msh.comm, "qtensor.xdmf",'w')
        xdmf_Q_file.write_mesh(msh)
        xdmf_Q_file.write_function(Q,0.0)
        print("Initial state written")
        xdmf_Q_file.close()

    # defining some constants
    A = fem.Constant(msh,PETSc.ScalarType(-0.064))
    B = fem.Constant(msh, PETSc.ScalarType(-1.57))
    C = fem.Constant(msh, PETSc.ScalarType(1.29))
    L = fem.Constant(msh, PETSc.ScalarType(1.0))
    k = fem.Constant(msh, PETSc.ScalarType(dt))

    # backwards euler part of residual
    F1 = ufl.inner((Q - Q_n)/k,V)*ufl.dx 
    # bulk free energy part
    F2 = ufl.inner((A*Q + B*ufl.dot(Q,Q) + C*(ufl.inner(Q,Q)*Q)),V)*ufl.dx
    #F2 = -1*ufl.inner((A*Q + B*ufl.dot(Q,Q) + C*(ufl.inner(Q,Q)*Q)),V)*ufl.dx
    # distortion/elastic term
    F3 = (ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
    #F3 = -1*(ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
    # construct the residual
    F = F1+F2+F3
    E = F2+F3
    #print(fem.assemble_scalar(F2+F3))
    print("Qvec sum",Q.vector.sum())
    print(FS.tabulate_dof_coordinates().shape)
    print(Q.vector.getValues(0))
    print(Q.vector.getArray()[0])
    #print(assemble(E))
    print(type(E))
    print(type(fem.form(E)))
    # Create nonlinear problem and Newton solver
    problem = fem.petsc.NonlinearProblem(F, Q)
    solver = nls.petsc.NewtonSolver(msh.comm, problem)
    solver.convergence_criterion = "residual" #"incremental"
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
    print("Init done")
    t = 0.0
    it = 0
    elapsed_time = 0
    Q_n.x.array[:] = Q.x.array[:]
    while (t < T):
        t += dt
        start_time = time.time()
        r = solver.solve(Q)
        Q.x.scatter_forward()
        Q_n.x.array[:] = Q.x.array[:] #swapping arrays
        if (isave):
            xdmf_Q_file.write_function(Q_n,t)
            #vtk_Q_file.write_function(Q_n,t)
        elapsed_time += time.time()-start_time
        #E = assemble(F2+F3)
        it += r[0]
        print(f"Step {int(t/dt)}/{nsteps}: num iterations: {r[0]}\t {elapsed_time}s,{it/elapsed_time}it/s")
    
    if (isave):
        xdmf_Q_file.close()
    print("Done!")
    print("Total time: ",elapsed_time)
    print("Total steps: ",nsteps)
    print("Total iterations: ",it)
    if (isave):
        print("converting Q-tensor to director field")
        xdmf_eig("qtensor.xdmf",nsteps)

    
if __name__ == '__main__':
    main()
