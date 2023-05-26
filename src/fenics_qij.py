# FEniCSx solver using TensorFunctionSpaces

import numpy as np
import ufl, time, meshio, gmsh
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc
from tqdm.auto import tqdm

nsteps = 50
dims = (0,0,0,1.0,1.0,0.5)
dt = 1e-3#T/nsteps
T = nsteps*dt
isave = False
debug = False
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
S0 = 0.53 # order parameter

def vtk_eig(filename,nsteps):
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points,cells = reader.read_points_cells()
        it = 0
        for k in tqdm(range(reader.num_steps)):
            t,point_data, cell_data = reader.read_data(k)
            data = point_data['f']
            eig_data=np.zeros((data.shape[0],3))
            Q_data = np.zeros((data.shape[0],3,3))
            for p in range(data.shape[0]):
                Q = np.reshape(data[p,:],(3,3))
                Q_data[p,:,:] = Q
                w,v = np.linalg.eig(Q)
                eig_data[p,:] = v[:,np.argmax(w)]
            new_mesh = meshio.Mesh(points,cells,point_data={"N": eig_data})
            vtk_filename = "director"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            new_mesh.write(vtk_filename)
            it += 1


def xdmf_eig(filename,nsteps):
    #reader = meshio.xdmf.TimeSeriesReader(filename)
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        with meshio.xdmf.TimeSeriesWriter("output.xdmf") as writer:
            writer.write_points_cells(points,cells)
            it = 0
            for k in tqdm(range(reader.num_steps)):
                t, point_data, cell_data = reader.read_data(k)
                data = point_data['f']
                eig_data = np.zeros((data.shape[0],3))
                Q = np.zeros((data.shape[0],3,3))
                for p in range(data.shape[0]):
                    iQ = np.reshape(data[p,:],(3,3))
                    Q[p,:,:] = iQ
                    w,v = np.linalg.eig(iQ) #w gives eigenvalues, v gives eigenvectors (v[:,i])
                    eig_data[p,:] = v[:,np.argmax(w)]

                #N_mesh = meshio.Mesh(points,cells,point_data={"N": eig_data,"Q":np.reshape(data,(data.shape[0],3,3))})
                #vtk_filename = "qtensor"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
                #N_mesh.write("output.xdmf")
                writer.write_data(t,point_data={"N":eig_data,"Q":np.reshape(data,(data.shape[0],3,3))})
                it += 1

def main():
    if (debug):
        log.set_log_level(log.LogLevel.INFO)

    gmsh.initialize()
    domain = gmsh.model.occ.addBox(dims[0],dims[1],dims[2],dims[3],dims[4],dims[5])
    gmsh.model.occ.synchronize()
    gdim = 3
    gmsh.model.addPhysicalGroup(gdim, [domain], 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",0.05)
    gmsh.model.mesh.generate(gdim)
    #gmsh.write("input_mesh.msh")
    msh,cell_markers,facet_markers = io.gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim=gdim)
    print("GMSH DONE")
    
    x = ufl.SpatialCoordinate(msh) 
    msh.topology.create_connectivity(msh.topology.dim-1,msh.topology.dim) #linking lists of cells/nodes/faces

    P = ufl.FiniteElement('Lagrange', msh.ufl_cell(), 1)
    FS_q0 = fem.FunctionSpace(msh,P) #CG == Lagrange
    FS_q1 = fem.FunctionSpace(msh,P)
    FS_q2 = fem.FunctionSpace(msh,P)
    FS_q3 = fem.FunctionSpace(msh,P)
    FS_q4 = fem.FunctionSpace(msh,P)

    EFS = fem.FunctionSpace(msh,("CG",1)) # function space for energy
    BFS = fem.FunctionSpace(msh,("CG",1)) # function space for biaxiality parameter

    print("nsteps:",nsteps)
    print("dt",dt)
    print("T:",T)

    # initial conditions
    def initqxx_rand(x):
        # values[0] = tensor[0,0]  --> 0 1 2
        # values[1] = tensor[0,1]      3 4 5
        # values[2] = tensor[0,2]      6 7 8
        # values[3] = tensor[1,0] ...
        values = np.zeros((3*3,x.shape[1]), dtype=np.float64)
        n = np.zeros((3,x.shape[1])) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x.shape[1]))
        azi_angle = np.random.uniform(0,2*np.pi,x.shape[1])
        n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        #n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        #n[2,:] = np.cos(polar_angle)
        return S0*(n[0,:]*n[0,:]-1/3)
    def initqyy_rand(x):
        # values[0] = tensor[0,0]  --> 0 1 2
        # values[1] = tensor[0,1]      3 4 5
        # values[2] = tensor[0,2]      6 7 8
        # values[3] = tensor[1,0] ...
        values = np.zeros((3*3,x.shape[1]), dtype=np.float64)
        n = np.zeros((3,x.shape[1])) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x.shape[1]))
        azi_angle = np.random.uniform(0,2*np.pi,x.shape[1])
        #n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        #n[2,:] = np.cos(polar_angle)
        return S0*(n[1,:]*n[1,:]-1/3)
    def initqxy_rand(x):
        # values[0] = tensor[0,0]  --> 0 1 2
        # values[1] = tensor[0,1]      3 4 5
        # values[2] = tensor[0,2]      6 7 8
        # values[3] = tensor[1,0] ...
        values = np.zeros((3*3,x.shape[1]), dtype=np.float64)
        n = np.zeros((3,x.shape[1])) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x.shape[1]))
        azi_angle = np.random.uniform(0,2*np.pi,x.shape[1])
        n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        #n[2,:] = np.cos(polar_angle)
        return S0*(n[0,:]*n[1,:])
    def initqxz_rand(x):
        # values[0] = tensor[0,0]  --> 0 1 2
        # values[1] = tensor[0,1]      3 4 5
        # values[2] = tensor[0,2]      6 7 8
        # values[3] = tensor[1,0] ...
        values = np.zeros((3*3,x.shape[1]), dtype=np.float64)
        n = np.zeros((3,x.shape[1])) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x.shape[1]))
        azi_angle = np.random.uniform(0,2*np.pi,x.shape[1])
        n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        #n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        n[2,:] = np.cos(polar_angle)
        return S0*(n[0,:]*n[2,:])
    def initqyz_rand(x):
        # values[0] = tensor[0,0]  --> 0 1 2
        # values[1] = tensor[0,1]      3 4 5
        # values[2] = tensor[0,2]      6 7 8
        # values[3] = tensor[1,0] ...
        values = np.zeros((3*3,x.shape[1]), dtype=np.float64)
        n = np.zeros((3,x.shape[1])) # director
        polar_angle = np.arccos(np.random.uniform(-1,1,x.shape[1]))
        azi_angle = np.random.uniform(0,2*np.pi,x.shape[1])
        #n[0,:] = np.sin(polar_angle)*np.cos(azi_angle)
        n[1,:] = np.sin(polar_angle)*np.sin(azi_angle)
        n[2,:] = np.cos(polar_angle)
        return S0*(n[1,:]*n[2,:])
    
    # Q = fem.Function(FS)
    # Q_n = fem.Function(FS)
    # V = fem.Function(FS)
    # (q0,q1,q2,q3,q4) = ufl.split(Q)
    # (q0_n,q1_n,q2_n,q3_n,q4_n) = ufl.split(Q_n)
    # (v0,v1,v2,v3,v4) = ufl.split(V)
    q0 = fem.Function(FS_q0) # current time-step result
    q1 = fem.Function(FS_q1)
    q2 = fem.Function(FS_q2)
    q3 = fem.Function(FS_q3)
    q4 = fem.Function(FS_q4)
    q0_n = fem.Function(FS_q0) # previous time-step result
    q1_n = fem.Function(FS_q1)
    q2_n = fem.Function(FS_q2)
    q3_n = fem.Function(FS_q3)
    q4_n = fem.Function(FS_q4)
    v0 = ufl.TestFunction(FS_q0) # test function to weight calcuations through the lattice
    v1 = ufl.TestFunction(FS_q1)
    v2 = ufl.TestFunction(FS_q2)
    v3 = ufl.TestFunction(FS_q3)
    v4 = ufl.TestFunction(FS_q4)

    # Q = fem.function(FS) # function for storing the result for output? 
    # Q.name = "Q"

#  Q = ufl.as_tensor(((q0_new, q4_new, q3_new),
#                        (q2_new, q1_new, q2_new),
#                        (q3_new, q4_new, -q0_new-q1_new)))
    #initializing Q for random director and distributing initial condition
    q0.interpolate(initqxx_rand)
    q0.x.scatter_forward()
    q1.interpolate(initqyy_rand)
    q1.x.scatter_forward()
    q2.interpolate(initqxy_rand)
    q2.x.scatter_forward()
    q3.interpolate(initqxz_rand)
    q3.x.scatter_forward()
    q4.interpolate(initqyz_rand)
    q4.x.scatter_forward()
    # setting up Dirichlet boundary conditions

    # setting up for all boundaries
    #boundary_facets = mesh.exterior_facet_indices(msh.topology)
    #boundary_dofs = fem.locate_dofs_topological(FS,msh.topology.dim-1,boundary_facets)
    #Q_bc.interpolate(initQ3d_anch)
    #bcs = [fem.dirichletbc(Q_bc,boundary_dofs)]

    # Q_bc_top = fem.Function(FS) # stores the boundary condition
    # Q_bc_bot = fem.Function(FS)
    # Q_bc_left = fem.Function(FS)
    # Q_bc_right = fem.Function(FS)
    # Q_bc_back = fem.Function(FS)
    # Q_bc_front = fem.Function(FS)

    # Q_bc_top.interpolate(initQ3d_anch)
    # Q_bc_bot.interpolate(initQ3d_defects)
    # Q_bc_left.interpolate(initQ3d_anch)
    # Q_bc_right.interpolate(initQ3d_anch)
    # Q_bc_front.interpolate(initQ3d_anch)
    # Q_bc_back.interpolate(initQ3d_anch)

    # dofs_top = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[2],dims[5])) # note: these only work for the unit cube
    # dofs_bot = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[2],dims[2]))
    # dofs_left = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[0],dims[0]))
    # dofs_right = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[0],dims[3]))
    # dofs_front = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[1],dims[1]))
    # dofs_back = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[1],dims[4]))
    # # X -> left_bc,right_bc
    # # Y -> front_bc,back_bc
    # # Z -> top_bc,bottom_bc
    # top_bc = fem.dirichletbc(Q_bc_top, dofs_top)
    # bottom_bc = fem.dirichletbc(Q_bc_bot, dofs_bot)
    # left_bc = fem.dirichletbc(Q_bc_left, dofs_left)
    # right_bc = fem.dirichletbc(Q_bc_right, dofs_right)
    # front_bc = fem.dirichletbc(Q_bc_front, dofs_front)
    # back_bc = fem.dirichletbc(Q_bc_back, dofs_back)

    # bcs = [top_bc,bottom_bc,left_bc,right_bc,front_bc,back_bc]

    # defining some constants
    # Zumer constants from Hydrodtnamics of pair-annihilating disclination lines in nematic liquid crystals
    # A = fem.Constant(msh,PETSc.ScalarType(-0.064))
    # B = fem.Constant(msh, PETSc.ScalarType(-1.57))
    # C = fem.Constant(msh, PETSc.ScalarType(1.29))
    # L = fem.Constant(msh, PETSc.ScalarType(1.0))
    #
    A = fem.Constant(msh,PETSc.ScalarType(-1))
    B = fem.Constant(msh, PETSc.ScalarType(-12.3))
    C = fem.Constant(msh, PETSc.ScalarType(10))
    L = fem.Constant(msh, PETSc.ScalarType(2.32))
    k = fem.Constant(msh, PETSc.ScalarType(dt))

    # q0 residual
    # backwards euler part of residual
    q0_f = ufl.inner(k*(A*q0_n + B*(q0_n*q0_n + q2_n*q2_n + q3_n*q3_n) + 2*C*q0_n*(q0_n*q0_n + q0_n*q1_n + q1_n*q1_n + q2_n*q2_n + q3_n*q3_n + q4_n*q4_n)),v0)*ufl.dx 
    q0_f += 1.0*ufl.inner(-q0,v0)*ufl.dx + ufl.inner(q0_n,v0)*ufl.dx - k*ufl.inner(ufl.grad(q0_n),ufl.grad(v0))*ufl.dx

    # q1 residual
    q1_f = ufl.inner(k*(A*q1_n + B*(q1_n*q1_n + q2_n*q2_n + q4_n*q4_n) + 2*C*q1_n*(q0_n*q0_n + q0_n*q1_n + q1_n*q1_n + q2_n*q2_n + q3_n*q3_n + q4_n*q4_n)),v1)*ufl.dx 
    q1_f += 1.0*ufl.inner(-q1,v1)*ufl.dx + ufl.inner(q1_n,v1)*ufl.dx - k*ufl.inner(ufl.grad(q1_n),ufl.grad(v1))*ufl.dx

    # q2 residual
    q2_f = ufl.inner(k*(A*q2_n + B*(q0_n*q2_n + q1_n*q2_n + q3_n*q4_n) + 2*C*q2_n*(q0_n*q0_n + q0_n*q1_n + q1_n*q1_n + q2_n*q2_n + q3_n*q3_n + q4_n*q4_n)),v2)*ufl.dx 
    q2_f += 1.0*ufl.inner(-q2,v2)*ufl.dx + ufl.inner(q2_n,v2)*ufl.dx - k*ufl.inner(ufl.grad(q2_n),ufl.grad(v2))*ufl.dx


    # q3 residual
    q3_f = ufl.inner(k*(A*q3_n + B*(-q1_n*q3_n + q2_n*q4_n) + 2*C*q3_n*(q0_n*q0_n + q0_n*q1_n + q1_n*q1_n + q2_n*q2_n + q3_n*q3_n + q4_n*q4_n)),v3)*ufl.dx 
    q3_f += 1.0*ufl.inner(-q3,v3)*ufl.dx + ufl.inner(q3_n,v3)*ufl.dx - k*ufl.inner(ufl.grad(q3_n),ufl.grad(v3))*ufl.dx

    # q4 residual
    q4_f = ufl.inner(k*(A*q4_n + B*(-q0_n*q4_n + q2_n*q3_n) + 2*C*q4_n*(q0_n*q0_n + q0_n*q1_n + q1_n*q1_n + q2_n*q2_n + q3_n*q3_n + q4_n*q4_n)),v4)*ufl.dx 
    q4_f += 1.0*ufl.inner(-q4,v4)*ufl.dx + ufl.inner(q4_n,v4)*ufl.dx - k*ufl.inner(ufl.grad(q4_n),ufl.grad(v4))*ufl.dx

    # #Creating excpression for the Frank Free energy
    # E_fn = fem.Expression(0.5*A*ufl.tr(Q*Q) + (B/3)*ufl.tr(Q*Q*Q) + 0.25*C*ufl.tr(Q*Q)*ufl.tr(Q*Q) + 0.5*L*ufl.inner(ufl.grad(Q),ufl.grad(Q)),EFS.element.interpolation_points())
    # E = fem.Function(EFS)
    # E.name = "E"
    # E.interpolate(E_fn)
    # prevE = np.sum(E.x.array[:])
    # print("Total Energy",prevE)

    # # biaxiality parameter
    # Biax_fn = fem.Expression(1 - 6*((ufl.tr(Q*Q*Q)**2)/(ufl.tr(Q*Q)**3)),BFS.element.interpolation_points())
    # Biax = fem.Function(BFS)
    # Biax.name = "Biax"
    # Biax.interpolate(Biax_fn)

    # writing initial conditions to file
    if (isave):
        xdmf_q0_file = io.XDMFFile(msh.comm, "qtensor_q0.xdmf",'w')
        xdmf_q0_file.write_mesh(msh)
        xdmf_q0_file.write_function(q0,0.0)

        xdmf_q1_file = io.XDMFFile(msh.comm, "qtensor_q1.xdmf",'w')
        xdmf_q1_file.write_mesh(msh)
        xdmf_q1_file.write_function(q1,0.0)

        xdmf_q2_file = io.XDMFFile(msh.comm, "qtensor_q2.xdmf",'w')
        xdmf_q2_file.write_mesh(msh)
        xdmf_q2_file.write_function(q2,0.0)

        xdmf_q3_file = io.XDMFFile(msh.comm, "qtensor_q3.xdmf",'w')
        xdmf_q3_file.write_mesh(msh)
        xdmf_q3_file.write_function(q3,0.0)

        xdmf_q4_file = io.XDMFFile(msh.comm, "qtensor_q4.xdmf",'w')
        xdmf_q4_file.write_mesh(msh)
        xdmf_q4_file.write_function(q4,0.0)

        # xdmf_E_file = io.XDMFFile(msh.comm, "energy.xdmf", 'w')
        # xdmf_E_file.write_mesh(msh)
        # xdmf_E_file.write_function(E,0.0)

        # xdmf_B_file = io.XDMFFile(msh.comm, "biaxiality.xdmf", 'w')
        # xdmf_B_file.write_mesh(msh)
        # xdmf_B_file.write_function(Biax,0.0)
        print("Initial state written")
        #xdmf_Q_file.close()

    # Create nonlinear problem and Newton solver
    problem_q0 = fem.petsc.NonlinearProblem(q0_f, q0)#, bcs)
    solver_q0 = nls.petsc.NewtonSolver(msh.comm, problem_q0)
    solver_q0.convergence_criterion = "incremental" #"residual"
    solver_q0.rtol = 1e-6
    ksp_q0 = solver_q0.krylov_solver
    opts_q0 = PETSc.Options()
    option_prefix = ksp_q0.getOptionsPrefix()
    opts_q0[f"{option_prefix}ksp_type"] = "preonly"
    opts_q0[f"{option_prefix}pc_type"] = "lu"
    ksp_q0.setFromOptions()

    problem_q1 = fem.petsc.NonlinearProblem(q1_f, q1)#, bcs)
    solver_q1 = nls.petsc.NewtonSolver(msh.comm, problem_q1)
    solver_q1.convergence_criterion = "incremental" #"incremental"
    solver_q1.rtol = 1e-6
    ksp_q1 = solver_q1.krylov_solver
    opts_q1 = PETSc.Options()
    option_prefix = ksp_q1.getOptionsPrefix()
    opts_q1[f"{option_prefix}ksp_type"] = "preonly"
    opts_q1[f"{option_prefix}pc_type"] = "lu"
    ksp_q1.setFromOptions()

    problem_q2 = fem.petsc.NonlinearProblem(q2_f, q2)#, bcs)
    solver_q2 = nls.petsc.NewtonSolver(msh.comm, problem_q2)
    solver_q2.convergence_criterion = "incremental" #"incremental"
    solver_q2.rtol = 1e-6
    ksp_q2 = solver_q2.krylov_solver
    opts_q2 = PETSc.Options()
    option_prefix = ksp_q2.getOptionsPrefix()
    opts_q2[f"{option_prefix}ksp_type"] = "preonly"
    opts_q2[f"{option_prefix}pc_type"] = "lu"
    ksp_q2.setFromOptions()

    problem_q3 = fem.petsc.NonlinearProblem(q3_f, q3)#, bcs)
    solver_q3 = nls.petsc.NewtonSolver(msh.comm, problem_q3)
    solver_q3.convergence_criterion = "incremental" #"incremental"
    solver_q3.rtol = 1e-6
    ksp_q3 = solver_q3.krylov_solver
    opts_q3 = PETSc.Options()
    option_prefix = ksp_q3.getOptionsPrefix()
    opts_q3[f"{option_prefix}ksp_type"] = "preonly"
    opts_q3[f"{option_prefix}pc_type"] = "lu"
    ksp_q3.setFromOptions()

    problem_q4 = fem.petsc.NonlinearProblem(q4_f, q4)#, bcs)
    solver_q4 = nls.petsc.NewtonSolver(msh.comm, problem_q4)
    solver_q4.convergence_criterion = "incremental" #"incremental"
    solver_q4.rtol = 1e-6
    ksp_q4 = solver_q4.krylov_solver
    opts_q4 = PETSc.Options()
    option_prefix = ksp_q4.getOptionsPrefix()
    opts_q4[f"{option_prefix}ksp_type"] = "preonly"
    opts_q4[f"{option_prefix}pc_type"] = "lu"
    ksp_q4.setFromOptions()

    #step in time
    print("Init done")
    t = 0.0
    it = 0
    istep = 0
    elapsed_time = 0
    elapsed_calc_time = 0
    elapsed_io_time = 0
    q0.x.array[:] = q0_n.x.array[:]
    q1.x.array[:] = q1_n.x.array[:]
    q2.x.array[:] = q2_n.x.array[:]
    q3.x.array[:] = q3_n.x.array[:]
    q4.x.array[:] = q4_n.x.array[:]
    while (t < T):
        t += dt
        istep += 1
        start_time = time.time()

        r_q0 = solver_q0.solve(q0)
        r_q1 = solver_q1.solve(q1)
        r_q2 = solver_q2.solve(q2)
        r_q3 = solver_q3.solve(q3)
        r_q4 = solver_q4.solve(q4)

        q0.x.scatter_forward()
        q1.x.scatter_forward()
        q2.x.scatter_forward()
        q3.x.scatter_forward()
        q4.x.scatter_forward()

        q0.x.array[:] = q0_n.x.array[:]
        q1.x.array[:] = q1_n.x.array[:]
        q2.x.array[:] = q2_n.x.array[:]
        q3.x.array[:] = q3_n.x.array[:]
        q4.x.array[:] = q4_n.x.array[:]
        # E.interpolate(E_fn)
        # Biax.interpolate(Biax_fn)
        # totalE = np.sum(E.x.array[:])
        it += r_q0[0] + r_q1[0] + r_q2[0] + r_q3[0] + r_q4[0]
        elapsed_calc_time += time.time() - start_time
        #if ((isave == True) and (int(t/dt)%10 == 0)):
        if (isave==True):
            io_start_time = time.time()
            xdmf_q0_file.write_function(q0_n,t)
            xdmf_q1_file.write_function(q1_n,t)
            xdmf_q2_file.write_function(q2_n,t)
            xdmf_q3_file.write_function(q3_n,t)
            xdmf_q4_file.write_function(q4_n,t)
            # xdmf_E_file.write_function(E,t)
            # xdmf_B_file.write_function(Biax,t)
            print("Saving at step ",int(t/dt))
            elapsed_io_time += time.time()-io_start_time
            #vtk_Q_file.write_function(Q_n,t)
        elapsed_time += time.time()-start_time
        if it/elapsed_time < 1.0:
            #print(f"Step {int(t/dt)}/{nsteps}:{r[0]} Total Energy:{round(totalE,3)} dE:{round(prevE-totalE,3)} {round(elapsed_time/it,2)}s/iter")
            print(f"Step {int(t/dt)}/{nsteps}:{r_q0[0] + r_q1[0] + r_q2[0] + r_q3[0] + r_q4[0]} {round(elapsed_time/it,2)}s/iter")
        else:
            #print(f"Step {int(t/dt)}/{nsteps}:{r[0]} Total Energy:{round(totalE,3)} dE:{round(prevE-totalE,3)} {round(it/elapsed_time,2)}iter/s")
            print(f"Step {int(t/dt)}/{nsteps}:{r_q0[0] + r_q1[0] + r_q2[0] + r_q3[0] + r_q4[0]} {round(it/elapsed_time,2)}iter/s")

       #prevE = totalE
    
    if (isave):
        #xdmf_Q_file.write_function(Q_n,T)
        xdmf_q0_file.close()
        xdmf_q1_file.close()
        xdmf_q2_file.close()
        xdmf_q3_file.close()
        xdmf_q4_file.close()
    print("Done!")
    print("Total time: ",elapsed_time)
    print("Calc: ",round((elapsed_calc_time/elapsed_time)*100,1),"% IO: ",round((elapsed_io_time/elapsed_time)*100,1),"%")
    print("Total steps: ",nsteps)
    print("Total iterations: ",it)
    if (isave):
        print("converting Q-tensor to director field")
        vtk_eig("qtensor_q0.xdmf",nsteps)
        vtk_eig("qtensor_q1.xdmf",nsteps)
        vtk_eig("qtensor_q2.xdmf",nsteps)
        vtk_eig("qtensor_q3.xdmf",nsteps)
        vtk_eig("qtensor_q4.xdmf",nsteps)

    
if __name__ == '__main__':
    main()
