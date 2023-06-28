import meshio
import numpy as np
import dolfinx.io
from mpi4py import MPI
from tqdm.auto import tqdm
import os

def setup(name,isave):
    if not os.path.exists("./data/"+name):
      os.makedirs("./data/"+name)
    else:
        print("sim directory exists")
    if os.listdir("./data/"+name):
        print("sim directory not empty")
        if isave[0] == True:
                exit()

def start_timers():
    return 0,0,0,0,0

def load_restart(filename):
    print(filename)
    xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD,filename,"r")
    msh = xdmf.read_mesh()
    xdmf.close()
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points,cells = reader.read_points_cells()
        t,point_data,cell_data = reader.read_data(reader.num_steps-1)
        Func_data = point_data['Q']
    return msh,Func_data

def vtk_eig(path,filename,nsteps):
    ''' function to read in tensor xdmf files and write director to individial vtk files'''
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
            vtk_filename = path+"director"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            new_mesh.write(vtk_filename)
            it += 1

def vtk_eig2(path,filename,nsteps):
    ''' function to read in tensor xdmf files and write director and scalar order paramter to 
    individual vtk files'''
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points,cells = reader.read_points_cells()
        it = 0
        for k in tqdm(range(reader.num_steps)):
            t,point_data,cell_data = reader.read_data(k)
            data = point_data['Q']
            eig_data = np.zeros((data.shape[0],3))
            Q_data = np.zeros((data.shape[0],3,3))
            S_data = np.zeros(data.shape[0])
            for p in range(data.shape[0]):
                Q = np.reshape(data[p,:],(3,3))
                Q_data[p,:,:] = Q
                w,v = np.linalg.eig(Q)
                n = v[:,np.argmax(w)]
                eig_data[p,:] = n
                S_data[p] = np.max(w)#1.5*np.dot(n,n)**2 - 0.5
            Q_mesh = meshio.Mesh(points,cells,point_data={"N": eig_data})
            vtk_filename = path+"director"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            Q_mesh.write(vtk_filename)
            S_mesh = meshio.Mesh(points,cells,point_data={"S":S_data})
            vtk_filename = path+"scalar"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            S_mesh.write(vtk_filename)
            it += 1

def vtk_biax(path,filename,nsteps):
    ''' functon to read in tensor xdmf files and write biaxiality parameter to individual vtk files'''
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points,cells = reader.read_points_cells()
        it = 0
        for k in tqdm(range(reader.num_steps)):
            t,point_data,cell_data = reader.read_data(k)
            data = point_data['Q']
            biax_data = np.zeros(data.shape[0])
            Q_data = np.zeros((data.shape[0],3,3))
            for p in range(data.shape[0]):
                Q = np.reshape(data[p,:],(3,3))
                Q_data[p,:,:] = Q
                #(1 - 6*((ufl.tr(self.Q*self.Q*self.Q)**2)/(ufl.tr(self.Q*self.Q)**3))
                Biax = 1 - 6*(np.trace(np.dot(np.dot(Q,Q),Q)**2)/np.trace(np.dot(Q,Q)**3))
                biax_data[p] = Biax
            new_mesh = meshio.Mesh(points,cells,point_data={"Biax": biax_data})
            vtk_filename = path+"biaxiality"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            new_mesh.write(vtk_filename)
            it += 1

def vtk_flatten_energy(path,filename):
    with meshio.xdmf. TimeSeriesReader(filename) as reader:
        points,cells = reader.read_points_cells()
        it = 0
        nsteps = reader.num_steps
        for k in tqdm(range(reader.num_steps)):   
            t,point_data,cell_data = reader.read_data(k)
            data = point_data['E']
            energy_data = np.zeros(data.shape[0])
            for p in range(data.shape[0]):
                Q = np.reshape(data[p,:],(3,3))
                Q_data[p,:,:] = Q
                #(1 - 6*((ufl.tr(self.Q*self.Q*self.Q)**2)/(ufl.tr(self.Q*self.Q)**3))
                Biax = 1 - 6*(np.trace((Q*Q*Q)**2)/np.trace((Q*Q)**3))
                #Biax = 1 - 6*(np.trace(np.dot(np.dot(Q,Q),Q)**2)/np.trace(np.dot(Q,Q)**3))
                biax_data[p] = Biax
            new_mesh = meshio.Mesh(points,cells,point_data={"Biax": biax_data})
            vtk_filename = path+"biaxiality"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
            new_mesh.write(vtk_filename)
            it += 1  

def xdmf_eig(filename,nsteps):
    ''' function to read in tensor xdmf files and write director to time series xdmf files '''
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