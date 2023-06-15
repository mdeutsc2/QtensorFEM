import meshio
import numpy as np
import dolfinx.io
from mpi4py import MPI
from tqdm.auto import tqdm

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

def vtk_eig(filename,nsteps):
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
            vtk_filename = "director"+str(it).zfill(len(str(nsteps))).replace('.','')+'.vtk'
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