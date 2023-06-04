# FEniCSx solver classes using TensorFunctionSpaces

import numpy as np
import ufl, time, meshio, gmsh
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc
from tqdm.auto import tqdm

# importing custom modules
from . import qij_io

class RelaxationQij3DTensor(object):
    '''Class to solve an initial'''

    # class constructor
    def __init__(self,dims,delta,nsteps,dt,
                      isave = (False,1),
                      debug=False,
                      restart=False):
        # PARAMETERS
        self.dims = dims
        self.delta = delta
        self.nsteps = nsteps
        self.dt = dt
        self.isave = isave
        self.debug = debug # defaults to false
        self.restart = restart

        # CLASS VARIABLES
        self.T = self.nsteps*self.dt
        self.msh = None # empty class variables
        self.FS = None
        self.energy_FS = None
        self.biax_FS = None
        self.Q = None
        self.Q_n = None
        self.V = None
        self.E = None
        self.Biax = None
        self.A = None
        self.B = None
        self.C = None
        self.L = None
        self.S0 = None
        self.bcs = None
        self.n_anch = None # kludge to get around no interpolation arguments
        self.w = None # defect spacing parameter (again, kludge)

    def initialize(self,A,B,C,L,S0):
        #----------------------
        # Initial Conditions
        #----------------------
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
            values[0] = self.S0*(n[0,:]*n[0,:]-1/3)
            values[1] = self.S0*(n[0,:]*n[1,:])
            values[2] = self.S0*(n[0,:]*n[2,:])
            values[3] = self.S0*(n[1,:]*n[0,:])
            values[4] = self.S0*(n[1,:]*n[1,:]-1/3)
            values[5] = values[3]
            values[6] = values[2]
            values[7] = values[1]
            values[8] = -values[0]-values[4]
            return values

        if (self.debug == True):
            log.set_log_level(log.LogLevel.INFO)
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        self.S0 = S0
        gmsh.initialize()
        domain = gmsh.model.occ.addBox(self.dims[0],self.dims[1],self.dims[2],self.dims[3],self.dims[4],self.dims[5])
        gmsh.model.occ.synchronize()
        gdim = 3
        gmsh.model.addPhysicalGroup(gdim, [domain], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",self.delta)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",self.delta)
        gmsh.model.mesh.generate(gdim)
        self.msh,cell_markers,facet_markers = io.gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim=gdim)
        print("-"*12,"GMSH DONE","-"*12,"\n")

        x = ufl.SpatialCoordinate(self.msh)
        self.msh.topology.create_connectivity(self.msh.topology.dim-1,self.msh.topology.dim) # linking lists of cells/nodes/faces

        # setting up Finite Element Spaces
        P = ufl.TensorElement('CG',self.msh.ufl_cell(),1,symmetry=True)
        self.FS = fem.FunctionSpace(self.msh,P)
        self.energy_FS = fem.FunctionSpace(self.msh,("CG",1)) # function space for energy
        self.biax_FS = fem.FunctionSpace(self.msh,("CG",1)) # function space for biaxiality parameter

        # setting up Functions
        self.Q = fem.Function(self.FS,name="Q")
        self.Q_n = fem.Function(self.FS,name="Q_n")
        self.V = ufl.TestFunction(self.FS)
        self.E = fem.Function(self.energy_FS,name="E")
        self.Biax = fem.Function(self.biax_FS,name="Biaxiality")

        #initializing Q for random director and distributing initial condition
        self.Q.interpolate(initQ3d_rand)
        self.Q.x.scatter_forward()

        print("=== PARAMETERS ===")
        print("nsteps:",self.nsteps)
        print("dt:",self.dt)
        print("T:",self.T)
        print("A",self.A,"B",self.B,"C",self.C,"L",self.L)
        print(self.Q.x.array[:].shape)
        print("DOF coords: ",self.FS.tabulate_dof_coordinates().shape)
        print("Global size: ",self.FS.dofmap.index_map.size_global)

    def run(self,boundary_conditions={}):
        #----------------------
        # Boundary Conditions
        #----------------------
        def initQ3d_anch(x,):
            values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
            n = np.zeros((3,x.shape[1])) # director
            n[0,:] = self.n_anch[0]
            n[1,:] = self.n_anch[1]
            n[2,:] = self.n_anch[2]
            values[0] = self.S0*(n[0,:]*n[0,:]-1/3)
            values[1] = self.S0*(n[0,:]*n[1,:])
            values[2] = self.S0*(n[0,:]*n[2,:])
            values[3] = self.S0*(n[1,:]*n[0,:])
            values[4] = self.S0*(n[1,:]*n[1,:]-1/3)
            values[5] = values[3]
            values[6] = values[2]
            values[7] = values[1]
            values[8] = -values[0]-values[4]
            return values

        def initQ3d_2defects(x):
            values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
            n = np.zeros((3,x.shape[1]))
            #theta = np.zeros((axispts,axispts))
            theta = 0.5*np.arctan2(x[1]-self.w/2,x[0]-0.25*self.w)-0.5*np.arctan2(x[1]-self.w/2,x[0]-0.75*self.w) + np.pi/2
            n[0,:] = np.cos(theta)
            n[1,:] = np.sin(theta)
            n[2,:] = 0.0
            values[0] = self.S0*(n[0,:]*n[0,:]-1/3)
            values[1] = self.S0*(n[0,:]*n[1,:])
            values[2] = self.S0*(n[0,:]*n[2,:])
            values[3] = self.S0*(n[1,:]*n[0,:])
            values[4] = self.S0*(n[1,:]*n[1,:]-1/3)
            values[5] = values[3]
            values[6] = values[2]
            values[7] = values[1]
            values[8] = -values[0]-values[4]
            return values

        def initQ3d_1defect(x):
            values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
            n = np.zeros((3,x.shape[1]))
            #theta = np.zeros((axispts,axispts))
            w = 2.5 # defect spacing
            theta = -0.5*np.arctan2(x[1]-self.w/2,x[0]-0.5*self.w) + np.pi
            #theta = 0.5*np.arctan2(x[1]-w/2,x[0]-0.25*w)-0.5*np.arctan2(x[1]-w/2,x[0]-0.75*w) + np.pi/2
            n[0,:] = np.cos(theta)
            n[1,:] = np.sin(theta)
            n[2,:] = 0.0
            values[0] = self.S0*(n[0,:]*n[0,:]-1/3)
            values[1] = self.S0*(n[0,:]*n[1,:])
            values[2] = self.S0*(n[0,:]*n[2,:])
            values[3] = self.S0*(n[1,:]*n[0,:])
            values[4] = self.S0*(n[1,:]*n[1,:]-1/3)
            values[5] = values[3]
            values[6] = values[2]
            values[7] = values[1]
            values[8] = -values[0]-values[4]
            return values

        # setting up for all boundaries
        # X -> left_bc,right_bc
        # Y -> front_bc,back_bc
        # Z -> top_bc,bottom_bc
        bcs_local = []
        if bool(boundary_conditions):
            self.bcs = boundary_conditions
            if 'top' in self.bcs:
                Q_bc_top = fem.Function(self.FS) # stores the boundary condition
                if 'defect' in self.bcs['top']:
                    if self.dims[3] != self.dims[4]:
                        print("substrate must be square for defects")
                        exit()
                    self.w = self.dims[3]
                    if self.bcs['top']['defect'] == 1:
                        Q_bc_top.interpolate(initQ3d_1defect)
                    elif self.bcs['top']['defect'] == 2:
                        Q_bc_top.interpolate(initQ3d_2defects)
                    else:
                        print("ndefects not specified correctly")
                        exit()
                elif 'anch' in self.bcs['top']:
                    self.n_anch = self.bcs['top']['anch']
                    Q_bc_top.interpolate(initQ3d_anch)

                top_bc = fem.dirichletbc(Q_bc_top, fem.locate_dofs_geometrical(self.FS, lambda x: np.isclose(x[2],self.dims[5])))
                bcs_local.append(top_bc)

            if 'bot' in self.bcs:
                Q_bc_bot = fem.Function(self.FS)
                if 'defect' in self.bcs['bot']:
                    if self.dims[3] != self.dims[4]:
                        print("substrate must be square for defects")
                        exit()
                    self.w = self.dims[3]
                    if self.bcs['bot']['defect'] == 1:
                        Q_bc_bot.interpolate(initQ3d_1defect)
                    elif self.bcs['bot']['defect'] == 2:
                        Q_bc_bot.interpolate(initQ3d_2defects)
                    else:
                        print("ndefects not specified correctly")
                        exit()
                elif 'anch' in self.bcs['bot']:
                    self.n_anch = self.bcs['bot']['anch']
                    Q_bc_bot.interpolate(initQ3d_anch)
                bottom_bc = fem.dirichletbc(Q_bc_bot, fem.locate_dofs_geometrical(self.FS, lambda x: np.isclose(x[2],self.dims[2])))
                bcs_local.append(bottom_bc)

            if 'left' in self.bcs:
                Q_bc_left = fem.Function(self.FS)
                if 'anch' in self.bcs['left']:
                    self.n_anch = self.bcs['left']['anch']
                    Q_bc_left.interpolate(initQ3d_anch)
                else:
                    print("defects not supported on left bc")
                left_bc = fem.dirichletbc(Q_bc_left, fem.locate_dofs_geometrical(self.FS, lambda x: np.isclose(x[0],self.dims[0])))
                bcs_local.append(left_bc)

            if 'right' in self.bcs:
                Q_bc_right = fem.Function(self.FS)
                if 'anch' in self.bcs['right']:
                    self.n_anch = self.bcs['right']['anch']
                    Q_bc_right.interpolate(initQ3d_anch)
                else:
                    print("defects not supported on right bc")
                right_bc = fem.dirichletbc(Q_bc_right, fem.locate_dofs_geometrical(self.FS, lambda x: np.isclose(x[0],self.dims[3])))
                bcs_local.append(right_bc)

            if 'back' in self.bcs:
                Q_bc_back = fem.Function(self.FS)
                if 'anch' in self.bcs['back']:
                    self.n_anch = self.bcs['back']['anch']
                    Q_bc_back.interpolate(initQ3d_anch)
                else:
                    print("defects not supported on back bc")
                back_bc = fem.dirichletbc(Q_bc_back, fem.locate_dofs_geometrical(self.FS, lambda x: np.isclose(x[1],self.dims[4])))
                bcs_local.append(back_bc)
            
            if 'front' in self.bcs:
                Q_bc_front = fem.Function(self.FS)
                if 'anch' in self.bcs['front']:
                    self.n_anch = self.bcs['front']['anch']
                    Q_bc_front.interpolate(initQ3d_anch)
                else:
                    print("defects not supported on front bc")
                front_bc = fem.dirichletbc(Q_bc_front, fem.locate_dofs_geometrical(self.FS, lambda x: np.isclose(x[1],self.dims[1])))
                bcs_local.append(front_bc)

            print("Supplied boundary conditions:",len(self.bcs))
            [print(item) for item in self.bcs.items()]
            print("FEniCSx boundary conditions:",len(bcs_local))
            if len(self.bcs) != len(bcs_local):
                print("boundary condition error")
                exit()
        else:
            print("Empty boundary conditions")
            exit()
        
        # defining some constants
        self.A = fem.Constant(self.msh,PETSc.ScalarType(self.A))
        self.B = fem.Constant(self.msh, PETSc.ScalarType(self.B))
        self.C = fem.Constant(self.msh, PETSc.ScalarType(self.C))
        self.L = fem.Constant(self.msh, PETSc.ScalarType(self.L))
        k = fem.Constant(self.msh, PETSc.ScalarType(self.dt))


        # backwards euler part of residual
        F1 = ufl.inner((self.Q - self.Q_n)/k,self.V)*ufl.dx 
        # bulk free energy part
        F2 = -1*ufl.inner((self.A*self.Q + self.B*ufl.dot(self.Q,self.Q) + self.C*(ufl.inner(self.Q,self.Q)*self.Q)),self.V)*ufl.dx
        # distortion/elastic term
        F3 = 0.5*self.L*(ufl.inner(ufl.grad(self.Q),ufl.grad(self.V)))*ufl.dx
        #F3 = -1*(ufl.inner(ufl.grad(Q),ufl.grad(V)))*ufl.dx
        # construct the residual
        F = F1+F2+F3

        #Creating excpression for the Frank Free energy
        E_fn = fem.Expression(0.5*self.A*ufl.tr(self.Q*self.Q) + 
                              (self.B/3)*ufl.tr(self.Q*self.Q*self.Q) + 
                              0.25*self.C*ufl.tr(self.Q*self.Q)*ufl.tr(self.Q*self.Q) + 
                              0.5*self.L*ufl.inner(ufl.grad(self.Q),ufl.grad(self.Q)),self.energy_FS.element.interpolation_points())
        self.E = fem.Function(self.energy_FS)
        self.E.name = "E"
        self.E.interpolate(E_fn)
        prevE = np.sum(self.E.x.array[:])
        print("Total Energy",prevE)

        # biaxiality parameter
        Biax_fn = fem.Expression(1 - 6*((ufl.tr(self.Q*self.Q*self.Q)**2)/(ufl.tr(self.Q*self.Q)**3)),self.biax_FS.element.interpolation_points())
        self.Biax = fem.Function(self.biax_FS)
        self.Biax.name = "Biax"
        self.Biax.interpolate(Biax_fn)

        # writing initial conditions to file
        if (self.isave[0]):
            xdmf_Q_file = io.XDMFFile(self.msh.comm, "qtensor.xdmf",'w')
            xdmf_Q_file.write_mesh(self.msh)
            xdmf_Q_file.write_function(self.Q,0.0)

            xdmf_E_file = io.XDMFFile(self.msh.comm, "energy.xdmf", 'w')
            xdmf_E_file.write_mesh(self.msh)
            xdmf_E_file.write_function(self.E,0.0)

            xdmf_B_file = io.XDMFFile(self.msh.comm, "biaxiality.xdmf", 'w')
            xdmf_B_file.write_mesh(self.msh)
            xdmf_B_file.write_function(self.Biax,0.0)
            print("Initial state written")

        # Create nonlinear problem and Newton solver
        problem = fem.petsc.NonlinearProblem(F, self.Q, bcs_local)
        solver = nls.petsc.NewtonSolver(self.msh.comm, problem)
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
        self.Q_n.x.array[:] = self.Q.x.array[:]
        while (t < self.T):
            t += self.dt
            istep += 1
            start_time = time.time()
            r = solver.solve(self.Q)
            self.Q.x.scatter_forward()
            self.Q_n.x.array[:] = self.Q.x.array #swapping old timestep for new timestep
            self.E.interpolate(E_fn)
            self.Biax.interpolate(Biax_fn)
            totalE = np.sum(self.E.x.array[:])
            it += r[0]
            elapsed_calc_time += time.time() - start_time
            if ((self.isave[0] == True) and (int(t/self.dt)%self.isave[1] == 0)):
                io_start_time = time.time()
                xdmf_Q_file.write_function(self.Q_n,t)
                xdmf_E_file.write_function(self.E,t)
                xdmf_B_file.write_function(self.Biax,t)
                print("Saving at step ",int(t/self.dt))
                elapsed_io_time += time.time()-io_start_time
                #vtk_Q_file.write_function(Q_n,t)
            elapsed_time += time.time()-start_time
            if it/elapsed_time < 1.0:
                print(f"Step {int(t/self.dt)}/{self.nsteps} It:{r[0]} Total Energy:{round(totalE,3)} dE:{round(prevE-totalE,3)} {round(elapsed_time/it,2)}s/iter")
            else:
                print(f"Step {int(t/self.dt)}/{self.nsteps} It:{r[0]} Total Energy:{round(totalE,3)} dE:{round(prevE-totalE,3)} {round(it/elapsed_time,2)}iter/s")
            prevE = totalE
        
        if (self.isave[0]):
            xdmf_Q_file.close()
            xdmf_E_file.close()
            xdmf_B_file.close()
        print("Done!")
        print("Total time: ",elapsed_time)
        print("Calc: ",round((elapsed_calc_time/elapsed_time)*100,1),"% IO: ",round((elapsed_io_time/elapsed_time)*100,1),"%")
        print("Total steps: ",self.nsteps)
        print("Total iterations: ",it)
        if (self.isave[0]):
            print("converting Q-tensor to director field")
            qij_io.vtk_eig("qtensor.xdmf",self.nsteps)