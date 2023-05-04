#q-tensor

#!/usr/bin/env python
import os,sys
if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import argparse
import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# custom modules
#from lbm_module import init_fluid,collide_stream,update_macro
#from lcfd_module import init_q,update_Q,calculate_stress

# note, for performace, debug=False
ti.init(arch=ti.cpu,default_fp=ti.f64,debug=True,device_memory_fraction=0.9)
show_Plot=False
save_vtk=True
#############################   
# Parameters
#############################
steps = 200
dt = 0.1
io_steps = 10
lx = 50
ly = 50
lz = 10
lbc = ti.field(dtype=int,shape=6) #liquid crystal boundary condition
lbc.from_numpy(np.array([0,0,0,0,0,0]))
niu = 0.67
Gamma = 0.625#0.33
gamma = 3#3.5
K = 0.55#0.0005
A0 = 1.0
S0 = 0.53
T = 0.5#0.6
eps = 0.59#0.8
ndefects = 4
xdefects = 2
ydefects = 2
xspacing = 80
yspacing = 80
defects = np.array([[-0.5,0.5],[0.5,-0.5]])

#############################
# Global Array Declarations
#############################
vel = ti.Vector.field(3,dtype=float,shape=(lx,ly,lz))
w = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz)) # velocity gradient tensor
wa = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
ws = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
S = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
Q = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz)) # Q-tensor
Q_new = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
Q_top_bc = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly))
Q_bot_bc = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly))
kdelta = ti.field(dtype=float,shape=(3,3))
kdelta.from_numpy(np.array([[1.0,0.0,0.0],
                            [0.0,1.0,0.0],
                            [0.0,0.0,1.0]]))
R = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz)) 
H = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
stress = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
stress_sym = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
stress_anti = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz))
dfdQ = ti.Matrix.field(n=3,m=3,dtype=float,shape=(lx,ly,lz)) # store dF/dQ for calculations, unsure if I can create a array withing ti.kernel

velgradx = ti.field(dtype=float,shape=ly)
velgradx.from_numpy(np.linspace(0.5,0.0,ly))

#############################
# General Routines
#############################
@ti.func
def ijk_bc(i:int,j:int,k:int,lx:int,ly:int,lz:int)->tuple[int,int,int,int,int,int]:
    ip1 = i + 1
    im1 = i - 1
    jp1 = j + 1
    jm1 = j - 1
    kp1 = k + 1
    km1 = k - 1
    # px,mx,py,my,pz,mz
    if ((ip1 > lx-1) and (lbc[0] == 0)):
        ip1 = 1
    if ((im1 < 0) and (lbc[1] == 0)):
        im1 = lx-1
    if ((jp1 > ly-1) and (lbc[2] == 0)):
        jp1 = 1
    if ((jm1 < 0) and (lbc[3] == 0)):
        jm1 = ly-1
    if ((kp1 > lz-1) and (lbc[4] == 0)):
        kp1 = 1
    if ((km1 < 0) and (lbc[5] == 0)):
        km1 = lz-1
    if ((kp1 > lz-1) and (lbc[4]==1)):
        kp1 = k
    if ((km1 < 0) and (lbc[5] == 1)):
        km1 = 0
    return ip1,im1,jp1,jm1,kp1,km1

#############################
# Liquid Crystal Routines
#############################
def init_lc():
    '''
    Initializes the Q-tensor field with random director using
        
        Q_ab = S_0 * (n_a * n_b - delta_ij/3)
        
        Parameters: S0 - the initial order parameter
    '''
    print("Init Q-tensor Field...")
    fill_vel()
    nx = np.empty((lx,ly,lz))
    ny = np.empty((lx,ly,lz))
    nz = np.empty((lx,ly,lz))
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                polar_angle = np.arccos(np.random.uniform(-1,1))
                azi_angle = np.random.uniform(0,2*np.pi)
                nx[i,j,k] = np.sin(polar_angle)*np.cos(azi_angle)
                ny[i,j,k] = np.sin(polar_angle)*np.sin(azi_angle)
                nz[i,j,k] = np.cos(polar_angle)

    if lbc[4] == 1: #+z boundary has defects
        print("generating +z defects")
        nx[:,:,-1],ny[:,:,-1],nz[:,:,-1] = gen_defects(lx,ly,lz,ndefects,xdefects,ydefects,xspacing,yspacing,defects)
    if lbc[5] == 1: #-z boundary has defects
        print("generating -z defects")
        nx[:,:,0],ny[:,:,0],nz[:,:,0] = gen_defects(lx,ly,lz,ndefects,xdefects,ydefects,xspacing,yspacing,defects)

    #gen defects throughout sample
    #for k in range(lz):
    #    nx[:,:,k],ny[:,:,k],nz[:,:,k] = gen_defects(lx,ly,lz,ndefects,xdefects,ydefects,xspacing,yspacing,defects)

    nx_field = ti.field(dtype=float,shape=(lx,ly,lz))
    nx_field.from_numpy(nx)
    ny_field = ti.field(dtype=float,shape=(lx,ly,lz))
    ny_field.from_numpy(ny)
    nz_field = ti.field(dtype=float,shape=(lx,ly,lz))
    nz_field.from_numpy(nz)
    gen_q(nx_field,ny_field,nz_field)

@ti.kernel
def fill_vel():
    for i,j, k in vel:
        for q in ti.static(range(3)):
            vel[i,j,k][q] = 0.0

@ti.kernel
def gen_q(nx:ti.template(),ny:ti.template(),nz:ti.template()):
    for i,j,k in Q:
        n = tm.vec3([0.0,0.0,0.0])
        n[0] = nx[i,j,k]
        n[1] = ny[i,j,k]
        n[2] = nz[i,j,k]
        #s = tm.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2])
        #n = n/s
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                Q[i,j,k][a,b] = S0*(n[a]*n[b] - kdelta[a,b]/3)
                if ((k == lz-1) and (lbc[4] == 1)):
                    Q_top_bc[i,j][a,b] = Q[i,j,k][a,b]
                if ((k == 0) and (lbc[5] == 1)):
                    Q_bot_bc[i,j][a,b] = Q[i,j,k][a,b]

def gen_defects(lx,ly,lz,ndefects,xdefects,ydefects,xspacing,yspacing,defects):
    nx = np.zeros((lx,ly))
    ny = np.zeros((lx,ly))
    nz = np.zeros((lx,ly))
    xx = np.zeros(ndefects)
    yy = np.zeros(ndefects)
    q = np.zeros(ndefects)
    th = np.zeros((lx,ly))
    id = 0
    if (ndefects >=1):
        for ii in range(1,xdefects+1):
            for jj in range(1,ydefects+1):
                if (xspacing != 0):
                    xx[id] = (ii*xspacing) + 0.5
                    xx[id] = xx[id] + (1.0-(xspacing*(xdefects+1.0))/lx)*(lx/2)
                else:
                    xx[id] = ii*(lx)/(xdefects+1.0)*0.5
                if (yspacing != 0):
                    yy[id] = (jj*yspacing) + 0.5
                    yy[id] = yy[id] + (1.0-(yspacing*(ydefects+1.0))/ly)*(ly/2)
                else:
                    yy[id] = jj*(ly)/(xdefects+1.0)*0.5
                q[id] = defects[ii-1,jj-1]
                id = id + 1
    else:
        id = 1
        xx[0] = lx/2 + 0.5
        yy[0] = ly/2 + 0.5

    for idefect in range(id):
        for i in range(lx):
            for j in range(ly):
                phi = np.arctan2((j+1)-yy[idefect],(i+1)-xx[idefect])
                #phi = np.arctan((j+1)-yy[idefect])/((i+1)-xx[idefect])
                th[i,j] += q[idefect]*phi + 0.25*np.pi
    
    for i in range(lx):
        for j in range(ly):
            nx[i,j] = 0.5*np.cos(th[i,j])
            ny[i,j] = 0.5*np.sin(th[i,j])
            nz[i,j] = 0.0
    return nx,ny,nz
           
@ti.kernel
def update_Q():
    Ko2 = 0.5*K
    for i,j,k in Q:
        ip1,im1,jp1,jm1,kp1,km1 = ijk_bc(i,j,k,lx,ly,lz)
        #calculating the velocity gradient
        w[i,j,k][0,0] = 0.5*(vel[ip1,j,k][0] - vel[im1,j,k][0])
        w[i,j,k][0,1] = 0.5*(vel[i,jp1,k][0] - vel[i,jm1,k][0])
        w[i,j,k][0,2] = 0.5*(vel[i,j,kp1][0] - vel[i,j,km1][0])
        w[i,j,k][1,0] = 0.5*(vel[ip1,j,k][1] - vel[im1,j,k][1])
        w[i,j,k][1,1] = 0.5*(vel[i,jp1,k][1] - vel[i,jm1,k][1])
        w[i,j,k][1,2] = 0.5*(vel[i,j,kp1][1] - vel[i,j,km1][1])
        w[i,j,k][2,0] = 0.5*(vel[ip1,j,k][2] - vel[im1,j,k][2])
        w[i,j,k][2,1] = 0.5*(vel[i,jp1,k][2] - vel[i,jm1,k][2])
        w[i,j,k][2,2] = 0.5*(vel[i,j,kp1][2] - vel[i,j,km1][2])
        #calculating the symmetric and anti-symmetric part of the velocity gradient tensor
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                ws[i,j,k][a,b] = 0.5*(w[i,j,k][a,b] + w[i,j,k][b,a])
                wa[i,j,k][a,b] = 0.5*(w[i,j,k][a,b] - w[i,j,k][b,a])

        # calculating the derivative of Landau-deGennes free energy
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                dfdQ[i,j,k][a,b] = 0.0
        # first bulk term of free energy, F
        C_b1 = A0*(1-(gamma/3))
        dfdQ[i,j,k][0,0] += C_b1*Q[i,j,k][0,0]
        dfdQ[i,j,k][0,1] += C_b1*Q[i,j,k][0,1]
        dfdQ[i,j,k][0,2] += C_b1*Q[i,j,k][0,2]
        dfdQ[i,j,k][1,1] += C_b1*Q[i,j,k][1,1]
        dfdQ[i,j,k][1,2] += C_b1*Q[i,j,k][1,2]
        dfdQ[i,j,k][2,2] += C_b1*Q[i,j,k][2,2]
        dfdQ[i,j,k][1,0] += C_b1*Q[i,j,k][0,1]
        dfdQ[i,j,k][2,1] += C_b1*Q[i,j,k][1,2]
        dfdQ[i,j,k][2,0] += C_b1*Q[i,j,k][0,2]
        # second bulk term of free energy,F
        C_b2 = -(A0*gamma)/3 # constant for b1 of Free energy term
        dfdQ[i,j,k][0,0] += C_b2*(3*Q[i,j,k][0,0]**2 + 3*Q[i,j,k][0,1]**2 + 3*Q[i,j,k][0,2]**2 + Q[i,j,k][0,1]*Q[i,j,k][1,1] + Q[i,j,k][0,2]*Q[i,j,k][1,2])
        dfdQ[i,j,k][0,1] += C_b2*(6*Q[i,j,k][0,0]*Q[i,j,k][0,1] + Q[i,j,k][0,0]*Q[i,j,k][1,1] + 4*Q[i,j,k][0,2]*Q[i,j,k][1,2] + 4*Q[i,j,k][0,1]*Q[i,j,k][1,1])
        dfdQ[i,j,k][0,2] += C_b2*(6*Q[i,j,k][0,0]*Q[i,j,k][0,2] + 4*Q[i,j,k][0,1]*Q[i,j,k][1,2] + Q[i,j,k][1,2]*Q[i,j,k][0,0] + 6*Q[i,j,k][0,2]*Q[i,j,k][2,2])
        dfdQ[i,j,k][1,1] += C_b2*(Q[i,j,k][0,1]*Q[i,j,k][0,0] + 2*Q[i,j,k][0,1]**2 + 2*Q[i,j,k][1,1]**2 + 2*Q[i,j,k][1,2]**2)
        dfdQ[i,j,k][1,2] += C_b2*(4*Q[i,j,k][0,1]*Q[i,j,k][0,2] + Q[i,j,k][0,2]*Q[i,j,k][0,0] + 4*Q[i,j,k][1,1]*Q[i,j,k][1,2] + 4*Q[i,j,k][1,2]*Q[i,j,k][2,2])
        dfdQ[i,j,k][2,2] += C_b2*(3*Q[i,j,k][0,2]**2 + 2*Q[i,j,k][1,2]**2 + 3*Q[i,j,k][2,2]**2)
        dfdQ[i,j,k][1,0] += C_b2*(6*Q[i,j,k][0,0]*Q[i,j,k][0,1] + Q[i,j,k][0,0]*Q[i,j,k][1,1] + 4*Q[i,j,k][0,2]*Q[i,j,k][1,2] + 4*Q[i,j,k][0,1]*Q[i,j,k][1,1])
        dfdQ[i,j,k][2,1] += C_b2*(4*Q[i,j,k][0,1]*Q[i,j,k][0,2] + Q[i,j,k][0,2]*Q[i,j,k][0,0] + 4*Q[i,j,k][1,1]*Q[i,j,k][1,2] + 4*Q[i,j,k][1,2]*Q[i,j,k][2,2])
        dfdQ[i,j,k][2,0] += C_b2*(6*Q[i,j,k][0,0]*Q[i,j,k][0,2] + 4*Q[i,j,k][0,1]*Q[i,j,k][1,2] + Q[i,j,k][1,2]*Q[i,j,k][0,0] + 6*Q[i,j,k][0,2]*Q[i,j,k][2,2])
        # third bulk term of free energy,F
        C_b3 = A0*gamma/4
        sumQ2 = Q[i,j,k][0,0]**2 + Q[i,j,k][0,1]**2 + Q[i,j,k][0,2]**2 + Q[i,j,k][1,1]**2 + Q[i,j,k][1,2]**2 + Q[i,j,k][2,2]**2
        dfdQ[i,j,k][0,0] += C_b3*2*sumQ2*2*Q[i,j,k][0,0]
        dfdQ[i,j,k][0,1] += C_b3*2*sumQ2*2*Q[i,j,k][0,1]
        dfdQ[i,j,k][0,2] += C_b3*2*sumQ2*2*Q[i,j,k][0,2]
        dfdQ[i,j,k][1,1] += C_b3*2*sumQ2*2*Q[i,j,k][1,1]
        dfdQ[i,j,k][1,2] += C_b3*2*sumQ2*2*Q[i,j,k][1,2]
        dfdQ[i,j,k][2,2] += C_b3*2*sumQ2*2*Q[i,j,k][2,2]
        dfdQ[i,j,k][1,0] += C_b3*2*sumQ2*2*Q[i,j,k][0,1]
        dfdQ[i,j,k][2,1] += C_b3*2*sumQ2*2*Q[i,j,k][1,2]
        dfdQ[i,j,k][2,0] += C_b3*2*sumQ2*2*Q[i,j,k][0,2]
        # distortion term of free energy,F
        dfdQ[i,j,k][0,0] += Ko2*(2*(Q[ip1,j,k][0,0]-Q[i,j,k][0,0]) + 2*(Q[im1,j,k][0,0]-Q[i,j,k][0,0])
                               + 2*(Q[i,jp1,k][0,0]-Q[i,j,k][0,0]) + 2*(Q[i,jm1,k][0,0]-Q[i,j,k][0,0])
                               + 2*(Q[i,j,kp1][0,0]-Q[i,j,k][0,0]) + 2*(Q[i,j,km1][0,0]-Q[i,j,k][0,0]))
        dfdQ[i,j,k][0,1] += Ko2*(2*(Q[ip1,j,k][0,1]-Q[i,j,k][0,1]) + 2*(Q[im1,j,k][0,1]-Q[i,j,k][0,1])
                               + 2*(Q[i,jp1,k][0,1]-Q[i,j,k][0,1]) + 2*(Q[i,jm1,k][0,1]-Q[i,j,k][0,1])
                               + 2*(Q[i,j,kp1][0,1]-Q[i,j,k][0,1]) + 2*(Q[i,j,km1][0,1]-Q[i,j,k][0,1]))
        dfdQ[i,j,k][0,2] += Ko2*(2*(Q[ip1,j,k][0,2]-Q[i,j,k][0,2]) + 2*(Q[im1,j,k][0,2]-Q[i,j,k][0,2])
                               + 2*(Q[i,jp1,k][0,2]-Q[i,j,k][0,2]) + 2*(Q[i,jm1,k][0,2]-Q[i,j,k][0,2])
                               + 2*(Q[i,j,kp1][0,2]-Q[i,j,k][0,2]) + 2*(Q[i,j,km1][0,2]-Q[i,j,k][0,2]))
        dfdQ[i,j,k][1,1] += Ko2*(2*(Q[ip1,j,k][1,1]-Q[i,j,k][1,1]) + 2*(Q[im1,j,k][1,1]-Q[i,j,k][1,1])
                               + 2*(Q[i,jp1,k][1,1]-Q[i,j,k][1,1]) + 2*(Q[i,jm1,k][1,1]-Q[i,j,k][1,1])
                               + 2*(Q[i,j,kp1][1,1]-Q[i,j,k][1,1]) + 2*(Q[i,j,km1][1,1]-Q[i,j,k][1,1]))
        dfdQ[i,j,k][1,2] += Ko2*(2*(Q[ip1,j,k][1,2]-Q[i,j,k][1,2]) + 2*(Q[im1,j,k][1,2]-Q[i,j,k][1,2])
                               + 2*(Q[i,jp1,k][1,2]-Q[i,j,k][1,2]) + 2*(Q[i,jm1,k][1,2]-Q[i,j,k][1,2])
                               + 2*(Q[i,j,kp1][1,2]-Q[i,j,k][1,2]) + 2*(Q[i,j,km1][1,2]-Q[i,j,k][1,2]))
        dfdQ[i,j,k][2,2] += Ko2*(2*(Q[ip1,j,k][2,2]-Q[i,j,k][2,2]) + 2*(Q[im1,j,k][2,2]-Q[i,j,k][2,2])
                               + 2*(Q[i,jp1,k][2,2]-Q[i,j,k][2,2]) + 2*(Q[i,jm1,k][2,2]-Q[i,j,k][2,2])
                               + 2*(Q[i,j,kp1][2,2]-Q[i,j,k][2,2]) + 2*(Q[i,j,km1][2,2]-Q[i,j,k][2,2]))
        dfdQ[i,j,k][1,0] = Ko2*(2*(Q[ip1,j,k][0,1]-Q[i,j,k][0,1]) + 2*(Q[im1,j,k][0,1]-Q[i,j,k][0,1])
                               + 2*(Q[i,jp1,k][0,1]-Q[i,j,k][0,1]) + 2*(Q[i,jm1,k][0,1]-Q[i,j,k][0,1])
                               + 2*(Q[i,j,kp1][0,1]-Q[i,j,k][0,1]) + 2*(Q[i,j,km1][0,1]-Q[i,j,k][0,1]))
        dfdQ[i,j,k][2,1] = Ko2*(2*(Q[ip1,j,k][1,2]-Q[i,j,k][1,2]) + 2*(Q[im1,j,k][1,2]-Q[i,j,k][1,2])
                               + 2*(Q[i,jp1,k][1,2]-Q[i,j,k][1,2]) + 2*(Q[i,jm1,k][1,2]-Q[i,j,k][1,2])
                               + 2*(Q[i,j,kp1][1,2]-Q[i,j,k][1,2]) + 2*(Q[i,j,km1][1,2]-Q[i,j,k][1,2]))
        dfdQ[i,j,k][2,0] = Ko2*(2*(Q[ip1,j,k][0,2]-Q[i,j,k][0,2]) + 2*(Q[im1,j,k][0,2]-Q[i,j,k][0,2])
                               + 2*(Q[i,jp1,k][0,2]-Q[i,j,k][0,2]) + 2*(Q[i,jm1,k][0,2]-Q[i,j,k][0,2])
                               + 2*(Q[i,j,kp1][0,2]-Q[i,j,k][0,2]) + 2*(Q[i,j,km1][0,2]-Q[i,j,k][0,2]))

        # resultant tensor for velocity dot grad Q
        R[i,j,k][0,0] = vel[i,j,k][0]*0.5*(Q[ip1,j,k][0,0]-Q[im1,j,k][0,0]) + vel[i,j,k][1]*0.5*(Q[i,jp1,k][0,0]-Q[i,jm1,k][0,0]) + vel[i,j,k][2]*0.5*(Q[i,j,kp1][0,0]-Q[i,j,km1][0,0])
        R[i,j,k][0,1] = vel[i,j,k][0]*0.5*(Q[ip1,j,k][0,1]-Q[im1,j,k][0,1]) + vel[i,j,k][1]*0.5*(Q[i,jp1,k][0,1]-Q[i,jm1,k][0,1]) + vel[i,j,k][2]*0.5*(Q[i,j,kp1][0,1]-Q[i,j,km1][0,1])
        R[i,j,k][0,2] = vel[i,j,k][0]*0.5*(Q[ip1,j,k][0,2]-Q[im1,j,k][0,2]) + vel[i,j,k][1]*0.5*(Q[i,jp1,k][0,2]-Q[i,jm1,k][0,2]) + vel[i,j,k][2]*0.5*(Q[i,j,kp1][0,2]-Q[i,j,km1][0,2])
        R[i,j,k][1,1] = vel[i,j,k][0]*0.5*(Q[ip1,j,k][1,1]-Q[im1,j,k][1,1]) + vel[i,j,k][1]*0.5*(Q[i,jp1,k][1,1]-Q[i,jm1,k][1,1]) + vel[i,j,k][2]*0.5*(Q[i,j,kp1][1,1]-Q[i,j,km1][1,1])
        R[i,j,k][1,2] = vel[i,j,k][0]*0.5*(Q[ip1,j,k][1,2]-Q[im1,j,k][1,2]) + vel[i,j,k][1]*0.5*(Q[i,jp1,k][1,2]-Q[i,jm1,k][1,2]) + vel[i,j,k][2]*0.5*(Q[i,j,kp1][1,2]-Q[i,j,km1][1,2])
        R[i,j,k][2,2] = vel[i,j,k][0]*0.5*(Q[ip1,j,k][2,2]-Q[im1,j,k][2,2]) + vel[i,j,k][1]*0.5*(Q[i,jp1,k][2,2]-Q[i,jm1,k][2,2]) + vel[i,j,k][2]*0.5*(Q[i,j,kp1][2,2]-Q[i,j,km1][2,2])
        R[i,j,k][1,0] = R[i,j,k][0,1]
        R[i,j,k][2,1] = R[i,j,k][1,2]
        R[i,j,k][2,0] = R[i,j,k][0,2]

        # compute S-tensor
        QWxx = Q[i,j,k][0,0]*w[i,j,k][0,0] + Q[i,j,k][0,1]*w[i,j,k][1,0] + Q[i,j,k][0,2]*w[i,j,k][2,0]
        QWyy = Q[i,j,k][1,0]*w[i,j,k][0,1] + Q[i,j,k][1,1]*w[i,j,k][1,1] + Q[i,j,k][1,2]*w[i,j,k][2,1]
        QWzz = Q[i,j,k][2,0]*w[i,j,k][0,2] + Q[i,j,k][2,1]*w[i,j,k][1,2] + Q[i,j,k][2,2]*w[i,j,k][2,2]
        TrQW = QWxx + QWyy + QWzz

        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                I = 0.0
                if a == b:
                    I = 1.0
                QpI = Q[i,j,k][a,b] + I*(1/3)
                S[i,j,k][a,b] = (eps*ws[i,j,k][a,b] + wa[i,j,k][a,b])*QpI + QpI*(eps*ws[i,j,k][a,b] + wa[i,j,k][a,b]) + 2*eps*QpI*TrQW

        # compute H-tensor
        # to compute H tensor, must compute dF/dQij
        #calculating the trace of dfdQ
        TrdfdQ = (dfdQ[i,j,k][0,0]+dfdQ[i,j,k][1,1] + dfdQ[i,j,k][2,2])/3
        H[i,j,k][0,0] = -1*dfdQ[i,j,k][0,0] + TrdfdQ
        H[i,j,k][0,1] = -1*dfdQ[i,j,k][0,1]
        H[i,j,k][0,2] = -1*dfdQ[i,j,k][0,2]
        H[i,j,k][1,0] = -1*dfdQ[i,j,k][1,0]
        H[i,j,k][1,1] = -1*dfdQ[i,j,k][1,1] + TrdfdQ
        H[i,j,k][1,2] = -1*dfdQ[i,j,k][1,2]
        H[i,j,k][2,0] = -1*dfdQ[i,j,k][2,0]
        H[i,j,k][2,1] = -1*dfdQ[i,j,k][2,1]
        H[i,j,k][2,2] = -1*dfdQ[i,j,k][2,2] + TrdfdQ

        # updating the new Q
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                Q_new[i,j,k][a,b] = dt*(Q[i,j,k][a,b] - R[i,j,k][a,b] + S[i,j,k][a,b] + Gamma*H[i,j,k][a,b])

@ti.kernel
def swap_Q():
    for i,j,k in Q:
        # swap Q
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                Q[i,j,k][a,b] = Q_new[i,j,k][a,b]
    #return Q,H,S

#############################
# I/O Routines
#############################

def write_lcd_vtk(istep,lx,ly,lz,Q):
    #write out director field and scalar order parameter to legacy VTK 2.0 files
    filename_lcs = "lcs"+str(istep).rjust(8,"0")+".vtk"
    filename_lcd = "lcd"+str(istep).rjust(8,"0")+".vtk"
    f_lcd = open(filename_lcd,"w+")
    f_lcd.write("# vtk DataFile Version 2.0\n")
    f_lcd.write(filename_lcd+"\n")
    f_lcd.write('ASCII\n')
    f_lcd.write('DATASET STRUCTURED_POINTS\n')
    f_lcd.write('DIMENSIONS '+str(lx)+" "+str(ly)+" "+str(lz)+"\n")
    f_lcd.write('ORIGIN 0 0 0\n')
    f_lcd.write('SPACING 1 1 1\n')
    f_lcd.write('POINT_DATA '+str(lx*ly*lz)+"\n")
    f_lcd.write('VECTORS Q_ab_director float\n')
    for k in range(Q.shape[2]):
        for j in range(Q.shape[1]):
            for i in range(Q.shape[0]):
                n = np.zeros(3) #corresponds to largest eigenvector of Q
                w,v = np.linalg.eig(Q[i,j,k,:,:]) #w gives eigenvalues, v gives eigenvectors (v[:,i])
                #print("eigenvalues",w)
                #print("eigenvectors",v)
                n = v[:,np.argmax(w)]
                #print("largest",n)
                f_lcd.write(str(np.real(n[0]))+" "+str(np.real(n[1]))+" "+str(np.real(n[2]))+"\n")
    f_lcd.close()

def write_biax(istep,lx,ly,lz,Q):
    filename_lcq = "lcq"+str(istep).rjust(8,"0")+".vtk"
    f_lcq = open(filename_lcq,"w+")
    f_lcq.write("# vtk DataFile Version 2.0\n")
    f_lcq.write(filename_lcq+"\n")
    f_lcq.write('ASCII\n')
    f_lcq.write('DATASET STRUCTURED_POINTS\n')
    f_lcq.write('DIMENSIONS '+str(lx)+" "+str(ly)+" "+str(lz)+"\n")
    f_lcq.write('ORIGIN 0 0 0\n')
    f_lcq.write('SPACING 1 1 1\n')
    f_lcq.write('POINT_DATA '+str(lx*ly*lz)+"\n")
    f_lcq.write('SCALARS biax_param float 1\n')
    f_lcq.write('LOOKUP_TABLE default\n')
    for k in range(Q.shape[2]):
        for j in range(Q.shape[1]):
            for i in range(Q.shape[0]):
                #Qi = np.zeros(3,3) #corresponds to largest eigenvector of Q
                Qi = Q[i,j,k,:,:]
                TrQi23 = np.trace(Qi*Qi)**3
                TrQi32 = np.trace(Qi*Qi*Qi)**2
                beta = 1-8*(TrQi32/TrQi23)
                f_lcq.write(str(beta)+"\n")
    f_lcq.close()
#############################
# MAIN
#############################
def main(args):
    start_time = time.time()
    init_lc()
    print("Init time: ",time.time()-start_time," s")
    elapsed_calc_time = 0.0
    elapsed_io_time =0.0
    if (save_vtk==True):
        print("Writing initial configurations...")
        write_lcd_vtk(0,lx,ly,lz,Q.to_numpy())
    print("Starting...")
    for istep in tqdm(range(steps)):
        start_calc_time = time.time()
        # collide and stream
        #collide_stream()

        # update_macro variables
        #update_macro()

        # update Q-tensor
        update_Q()
        swap_Q()            
        #lc_bc()

        #compute stress tensor using Q-tensor and H-tensor
        #calculate_stress()

        #i/o
        elapsed_calc_time += time.time()-start_calc_time
        if ((istep+1)%io_steps == 0) and (save_vtk == True):
            start_io_time = time.time()
            tqdm.write("writing out data at step %i" % (istep+1))
            veln = vel.to_numpy()
            #write_fluid_vtk(i,params,veln)
            Qn = Q.to_numpy().astype(np.float32)
            write_lcd_vtk(istep+1,lx,ly,lz,Qn)
            #write_biax(istep+1,lx,ly,lz,Qn)
            elapsed_io_time += time.time()-start_io_time
    total_time = time.time()-start_time
    print("Total time: ",time.strftime("%H:%M:%S", time.gmtime(total_time)))
    print("Computation time: ",elapsed_calc_time," s\t",elapsed_calc_time/total_time)
    print("I/O time: ",elapsed_io_time," s\t",elapsed_io_time/total_time)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--name", action="store", dest="name")

    args = parser.parse_args()
    main(args)