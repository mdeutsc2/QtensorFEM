# FEniCSx solver using TensorFunctionSpaces

import numpy as np
import ufl, time, meshio, gmsh
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc
from tqdm.auto import tqdm

nsteps = 50
dims = (0,0,0,1.0,1.0,0.5)
dt = 1e-4#T/nsteps
T = nsteps*dt
isave = True
debug = False
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
S0 = 0.53 # order parameter

def vtk_eig(filename,nsteps):
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points,cells = reader.read_points_cells()
        it = 0
        for k in tqdm(range(reader.num_steps)):
            t,point_data, cell_data = reader.read_data(k)
            data = point_data['Q']
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
    print("-"*12,"GMSH DONE","-"*12)
    
    x = ufl.SpatialCoordinate(msh) 
    msh.topology.create_connectivity(msh.topology.dim-1,msh.topology.dim) #linking lists of cells/nodes/faces
    #P = ufl.TensorElement('CG', ufl.tetrahedron, 1, symmetry=True)
    P = ufl.TensorElement('CG',msh.ufl_cell(),1,symmetry=True)
    FS = fem.FunctionSpace(msh,P) #CG == Lagrange
    EFS = fem.FunctionSpace(msh,("CG",4)) # function space for energy
    BFS = fem.FunctionSpace(msh,("CG",4)) # function space for biaxiality parameter

    #P2 = ufl.VectorElement('CG',msh.ufl_cell(),1)
    #EFS = fem.VectorFunctionSpace(msh,P)
    print("nsteps:",nsteps)
    print("dt",dt)
    print("T:",T)
    # initial conditions
    def initQ3d_rand(x):
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

    def initQ3d_anch(x):
        values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
        n = np.zeros((3,x.shape[1])) # director
        n[0,:] = 0.0
        n[1,:] = 1.0
        n[2,:] = 0.0
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

    def initQ3d_defects(x):
        values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
        n = np.zeros((3,x.shape[1]))
        #theta = np.zeros((axispts,axispts))
        w = 1.0 # defect spacing
        theta = 0.5*np.arctan2(x[1]-w/2,x[0]-0.25*w)-0.5*np.arctan2(x[1]-w/2,x[0]-0.75*w) + np.pi/2
        n[0,:] = np.cos(theta)
        n[1,:] = np.sin(theta)
        n[2,:] = 0.0
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
    
    Q = fem.Function(FS) # current time-step result
    Q.name = "Q"
    Q_n = fem.Function(FS) # previous time-step result
    Q_n.name = "Q"
    V = ufl.TestFunction(FS) # test function to weight calcuations through the lattice
    #E = fem.Function(EFS) # energy function

    #initializing Q for random director and distributing initial condition
    Q.interpolate(initQ3d_rand)
    Q.x.scatter_forward()
    print(Q.x.array[:].shape)
    #print("len(x.array[:] ", len(Q.x.array[:]))
    print("DOF coords: ",FS.tabulate_dof_coordinates().shape)
    print("Global size: ",FS.dofmap.index_map.size_global)
    # setting up Dirichlet boundary conditions

    # setting up for all boundaries
    #boundary_facets = mesh.exterior_facet_indices(msh.topology)
    #boundary_dofs = fem.locate_dofs_topological(FS,msh.topology.dim-1,boundary_facets)
    #Q_bc.interpolate(initQ3d_anch)
    #bcs = [fem.dirichletbc(Q_bc,boundary_dofs)]

    Q_bc_top = fem.Function(FS) # stores the boundary condition
    Q_bc_bot = fem.Function(FS)
    Q_bc_left = fem.Function(FS)
    Q_bc_right = fem.Function(FS)
    Q_bc_back = fem.Function(FS)
    Q_bc_front = fem.Function(FS)

    Q_bc_top.interpolate(initQ3d_anch)
    Q_bc_bot.interpolate(initQ3d_defects)
    Q_bc_left.interpolate(initQ3d_anch)
    Q_bc_right.interpolate(initQ3d_anch)
    Q_bc_front.interpolate(initQ3d_anch)
    Q_bc_back.interpolate(initQ3d_anch)

    dofs_top = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[2],dims[5])) # note: these only work for the unit cube
    dofs_bot = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[2],dims[2]))
    dofs_left = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[0],dims[0]))
    dofs_right = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[0],dims[3]))
    dofs_front = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[1],dims[1]))
    dofs_back = fem.locate_dofs_geometrical(FS, lambda x: np.isclose(x[1],dims[4]))
    # X -> left_bc,right_bc
    # Y -> front_bc,back_bc
    # Z -> top_bc,bottom_bc
    top_bc = fem.dirichletbc(Q_bc_top, dofs_top)
    bottom_bc = fem.dirichletbc(Q_bc_bot, dofs_bot)
    left_bc = fem.dirichletbc(Q_bc_left, dofs_left)
    right_bc = fem.dirichletbc(Q_bc_right, dofs_right)
    front_bc = fem.dirichletbc(Q_bc_front, dofs_front)
    back_bc = fem.dirichletbc(Q_bc_back, dofs_back)

    bcs = [top_bc,bottom_bc,left_bc,right_bc,front_bc,back_bc]
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


    # backwards euler part of residual
    F1 = ufl.inner((Q - Q_n)/k,V)*ufl.dx 
    # bulk free energy part
    F2 = -1*ufl.inner((A*Q + B*ufl.dot(Q,Q) + C*(ufl.inner(Q,Q)*Q)),V)*ufl.dx
    # distortion/elastic term
    F3 = (ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
    #F3 = -1*(ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
    # construct the residual
    F = F1+F2+F3

    #Creating excpression for the Frank Free energy
    E_fn = fem.Expression(0.5*A*ufl.tr(Q*Q) + (B/3)*ufl.tr(Q*Q*Q) + 0.25*C*ufl.tr(Q*Q)*ufl.tr(Q*Q) + 0.5*L*ufl.inner(ufl.grad(Q),ufl.grad(Q)),EFS.element.interpolation_points())
    E = fem.Function(EFS)
    E.name = "E"
    E.interpolate(E_fn)
    prevE = np.sum(E.x.array[:])
    print("Total Energy",prevE)

    # biaxiality parameter
    Biax_fn = fem.Expression(1 - 6*((ufl.tr(Q*Q*Q)**2)/(ufl.tr(Q*Q)**3)),BFS.element.interpolation_points())
    Biax = fem.Function(BFS)
    Biax.name = "Biax"
    Biax.interpolate(Biax_fn)

    # writing initial conditions to file
    if (isave):
        xdmf_Q_file = io.XDMFFile(msh.comm, "qtensor.xdmf",'w')
        xdmf_Q_file.write_mesh(msh)
        xdmf_Q_file.write_function(Q,0.0)

        xdmf_E_file = io.XDMFFile(msh.comm, "energy.xdmf", 'w')
        xdmf_E_file.write_mesh(msh)
        xdmf_E_file.write_function(E,0.0)

        xdmf_B_file = io.XDMFFile(msh.comm, "biaxiality.xdmf", 'w')
        xdmf_B_file.write_mesh(msh)
        xdmf_B_file.write_function(Biax,0.0)
        print("Initial state written")
        #xdmf_Q_file.close()

    # Create nonlinear problem and Newton solver
    problem = fem.petsc.NonlinearProblem(F, Q, bcs)
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
    istep = 0
    elapsed_time = 0
    elapsed_calc_time = 0
    elapsed_io_time = 0
    Q_n.x.array[:] = Q.x.array[:]
    while (t < T):
        t += dt
        istep += 1
        start_time = time.time()
        r = solver.solve(Q)
        Q.x.scatter_forward()
        Q_n.x.array[:] = Q.x.array #swapping old timestep for new timestep
        E.interpolate(E_fn)
        Biax.interpolate(Biax_fn)
        totalE = np.sum(E.x.array[:])
        it += r[0]
        elapsed_calc_time += time.time() - start_time
        #if ((isave == True) and (int(t/dt)%10 == 0)):
        if (isave==True):
            io_start_time = time.time()
            xdmf_Q_file.write_function(Q_n,t)
            xdmf_E_file.write_function(E,t)
            xdmf_B_file.write_function(Biax,t)
            print("Saving at step ",int(t/dt))
            elapsed_io_time += time.time()-io_start_time
            #vtk_Q_file.write_function(Q_n,t)
        elapsed_time += time.time()-start_time
        if it/elapsed_time < 1.0:
            print(f"Step {int(t/dt)}/{nsteps}:{r[0]} Total Energy:{round(totalE,3)} dE:{round(prevE-totalE,3)} {round(elapsed_time/it,2)}s/iter")
        else:
            print(f"Step {int(t/dt)}/{nsteps}:{r[0]} Total Energy:{round(totalE,3)} dE:{round(prevE-totalE,3)} {round(it/elapsed_time,2)}iter/s")
        prevE = totalE
    
    if (isave):
        #xdmf_Q_file.write_function(Q_n,T)
        xdmf_Q_file.close()
        xdmf_E_file.close()
        xdmf_B_file.close()
    print("Done!")
    print("Total time: ",elapsed_time)
    print("Calc: ",round((elapsed_calc_time/elapsed_time)*100,1),"% IO: ",round((elapsed_io_time/elapsed_time)*100,1),"%")
    print("Total steps: ",nsteps)
    print("Total iterations: ",it)
    if (isave):
        print("converting Q-tensor to director field")
        vtk_eig("qtensor.xdmf",nsteps)

    
if __name__ == '__main__':
    main()
