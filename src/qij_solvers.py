# FEniCSx solver classes using TensorFunctionSpaces

import numpy as np
import ufl, time, meshio, gmsh
from dolfinx import *
from mpi4py import MPI
from petsc4py import PETSc
from tqdm.auto import tqdm

# importing custom modules

class RelaxationQij3D:
    '''Class to solver an initial'''

    # class constructor
    def __init__(self,dims,pts,nsteps,dt,
                      isave = (False,1),
                      debug=False):
        # PARAMETERS
        self.dims = dims
        self.pts = pts
        self.nsteps = nsteps
        self.dt = dt
        self.isave = isave
        self.debug = debug # defaults to false

        # CLASS VARIABLES
        self.T = self.nsteps*self.dt
        self.mesh = None # empty class variables
        self.FS = None
        self.energy_FS = None
        self.biax_FS = None
        self.Q = None
        self.Q_n = None
        self.V = None
        self.E = None
        self.Biax = None

    def initialize(self):
        if (self.debug == True):
            log.set_log_level(log.LogLevel.INFO)

        gmsh.initialize()
        domain = gmsh.model.occ.addBox(self.dims[0],self.dims[1],self.dims[2],self.dims[3],self.dims[4],self.dims[5])
        gmsh.model.occ.synchronize()
        gdim = 3
        delta = np.max([self.dims[3]/self.pts[0],self.dims[4]/self.pts[1],self.dims[5]/self.pts[2]])
        gmsh.model.addPhysicalGroup(gdim, [domain], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",delta)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",delta)
        gmsh.model.mesh.generate(gdim)
        self.msh,cell_markers,facet_markers = io.gmshio.model_to_mesh(gmsh.model,MPI.COMM_WORLD,0,gdim=gdim)
        print("GMSH DONE")

        x = ufl.SpatialCoordinate(self.msh)
        self.msh.topology.create_connectivity(self.msh.topology.dim-1,self.msh.topology.dim) # linking lists of cells/nodes/faces

        # setting up Finite Element Spaces
        P = ufl.TensorElement('CG',self.msh.ufl_cell(),2,symmetry=True)
        self.FS = fem.FunctionSpace(self.msh,P)
        self.energy_FS = fem.FunctionSpace(self.msh,("CG",4)) # function space for energy
        self.biax_FS = fem.FunctionSpace(self.msh,("CG",4)) # function space for biaxiality parameter

        # setting up Functions
        self.Q = fem.Function(self.FS,name="Q")
        self.Q_n = fem.Function(self.FS,name="Q_n")
        self.V = ufl.TestFunction(self.FS)
        self.E = fem.Function(self.energy_FS,name="E")
        self.Biax = fem.Function(self.biax_FS,name="Biaxiality")


        print("=== PARAMETERS ===")
        print("nsteps:",self.nsteps)
        print("dt:",self.dt)
        print("T:",self.T)
        print(self.Q.x.array[:].shape)
        print("DOF coords: ",self.FS.tabulate_dof_coordinates().shape)
        print("Global size: ",self.FS.dofmap.index_map.size_global)

    def solver(self):
        pass

dims = (0,0,0,1,1,1) #lower x,y,z upper x,y,z
pts = (10,10,10) #number of points in each direction
nsteps = 100
dt = 1e-6
isave = (True,int(nsteps/10))
S0 = 0.53
A = -0.064
B = -1.57
C = 1.29
L = 1.0
sim = RelaxationQij3D(dims,pts,nsteps,dt,isave)#,S0,A,B,C,L)
sim.initialize()