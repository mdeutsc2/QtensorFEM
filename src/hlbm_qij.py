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
fbc = ti.field(dtype=int,shape=6) #fluid boundary conditions [+x,-x,+y,-y,+z,-z]
fbc.from_numpy(np.array([0,0,0,0,0,0]))

fbc_values = ti.field(dtype=float,shape=(6,3))
fbc_values.from_numpy(np.array([[0.0,0.0,0.0],
                                [0.0,0.0,0.0],
                                [0.0,0.0,0.0],
                                [0.0,0.0,0.0],
                                [0.0,0.0,0.0],
                                [0.0,0.0,0.0]]))
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
rho = ti.field(dtype=float,shape=(lx,ly,lz))
vel = ti.Vector.field(3,dtype=float,shape=(lx,ly,lz))
f_old = ti.Vector.field(15,dtype=float,shape=(lx,ly,lz))
f_new = ti.Vector.field(15,dtype=float,shape=(lx,ly,lz))
f_eq = ti.Vector.field(15,dtype=float,shape=(lx,ly,lz))
w = ti.field(dtype=float, shape=19)
e = ti.field(dtype=float, shape=(19,3))
w.from_numpy(np.array([ 1.0/3.0, # i = 0
                        1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, # i= 1->6 
                        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0], 
                        dtype=float))
e.from_numpy(np.array([[0, 0, 0],
                       [1, 0,0 ], 
                       [-1, 0, 0], 
                       [0, 1, 0], 
                       [0, -1, 0], 
                       [0, 0, 1],
                       [0, 0, -1], 
                       [1, 1, 0], 
                       [-1, -1, 0],
                       [1, -1, 0],
                       [-1, 1, 0],
                       [1, 0, 1],
                       [-1, 0, -1],
                       [1, 0, -1],
                       [-1, 0, -1],
                       [0, 1, 1],
                       [0, -1, -1],
                       [0, 1, -1],
                       [0, -1, 1]], dtype=np.int64))
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
# Lattice Boltzmann Routines
#############################
@ti.kernel
def init_lb():
    print("Init fluid fields")
    for i,j,k in rho:
        vel[i,j,k][0] = 0.0
        vel[i,j,k][1] = 0.0#velgradx[i]
        vel[i,j,k][2] = 0.0
        rho[i,j,k] = 1.0
        # for c in ti.static(range(15)):
        #     f_eq[i,j,k][c] = f_eq(i,j,k,T)
        #     f_new[i,j,k][c] = feq[i,j,k][c]
        #     f_old[i,j,k][c] = f_new[i,j,k][c]

@ti.func
def f_eq(i: int,j: int,k: int,T:float)->float:
    feq = 0.0
    TrQo3 = (stress[i,j,k][1,1]/3 + stress[i,j,k][2,2]/3 + stress[i,j,k][3,3]/3)
    v2 = vel[i,j,k][0]*vel[i,j,k][0] + vel[i,j,k][1]*vel[i,j,k][1] + vel[i,j,k][2]*vel[i,j,k][2]
    for q in ti.static(range(15)):
        ve = vel[i,j,k][0]*e[q,0] + vel[i,j,k][1]*e[q,1] + vel[i,j,k][2]*e[q,2]
        vvee = vel[i,j,k][0]*vel[i,j,k][0]*e[q,0]*e[q,0] + vel[i,j,k][0]*vel[i,j,k][1]*e[q,0]*e[q,1] +\
               vel[i,j,k][0]*vel[i,j,k][2]*e[q,0]*e[q,2] + vel[i,j,k][1]*vel[i,j,k][1]*e[q,1]*e[q,1] +\
               vel[i,j,k][1]*vel[i,j,k][2]*e[q,1]*e[q,2] + vel[i,j,k][2]*vel[i,j,k][2]*e[q,2]*e[q,2]
        if (q == 0):
            #A = rho[i,j,k] + 1.4*TrQo3
            A = rho[i,j,k] - 1.4*rho[i,j,k]*T
            C = -2*rho[i,j,k]/3
            feq = A + C*v2

        if (q >= 1 or q <= 6):
            #A = -0.1*TrQo3
            A = 0.1*rho[i,j,k]*T
            B = rho[i,j,k]/3
            C = -0.5*rho[i,j,k]
            D = 0.5*rho[i,j,k]
            Exx = 0.5*(-1.0*stress[i,j,k][1,1] + TrQo3)*e[q,0]*e[q,0]
            Exy = 0.5*(-1.0*stress[i,j,k][1,2])*e[q,0]*e[q,1]
            Exz = 0.5*(-1.0*stress[i,j,k][1,3])*e[q,0]*e[q,2]
            Eyy = 0.5*(-1.0*stress[i,j,k][2,2] + TrQo3)*e[q,1]*e[q,1]
            Eyz = 0.5*(-1.0*stress[i,j,k][2,3])*e[q,1]*e[q,2]
            Ezz = 0.5*(-1.0*stress[i,j,k][3,3] + TrQo3)*e[q,2]*e[q,2]
            E = Exx+Exy+Exz+Eyy+Eyz+Ezz
            f_eq = A + B*ve + C*u2 + D*vvee + E
        if (q>=7):
            #A = -0.1*TrQo3
            A = 0.1*rho[i,j,k]*T
            B = rho[i,j,k]/24
            C = -rho[i,j,k]/24
            D = rho[i,j,k]/16
            Exx = 0.0625*(-1.0*stress[i,j,k][1,1] + TrQo3)*e[q,0]*e[q,0]
            Exy = 0.0625*(-1.0*stress[i,j,k][1,2])*e[q,0]*e[q,1]
            Exz = 0.0625*(-1.0*stress[i,j,k][1,3])*e[q,0]*e[q,2]
            Eyy = 0.0625*(-1.0*stress[i,j,k][2,2] + TrQo3)*e[q,1]*e[q,1]
            Eyz = 0.0625*(-1.0*stress[i,j,k][2,3])*e[q,1]*e[q,2]
            Ezz = 0.0625*(-1.0*stress[i,j,k][3,3] + TrQo3)*e[q,2]*e[q,2]
            E = Exx+Exy+Exz+Eyy+Eyz+Ezz
            f_eq = A+B*ve+C*u2+D*vvee+E
    return feq

@ti.func
def collision():
    # TODO
    pass

@ti.kernel
def collide_stream():
    for i,j,k in ti.ndrange((1,nx-1),(1,ny-1),(1,nz-1)):
        for q in ti.static(range(15)):
            ip = i - ti.cast(e[k,0],ti.int32)
            jp = j - ti.cast(e[k,1],ti.int32)
            kp = k - ti.cast(e[k,2],ti.int32)
            f_new[i,j,k][q] = f_old[i,j,k][q] + collision()
    #return f_new

@ti.kernel
def update_macro():
    for i,j,k in ti.ndrange((1,params.nx-1),(1,params.ny-1),(1,params.nz-1)):
        rho[i,j,k] = 0.0
        vel[i,j,k][0] = 0.0
        vel[i,j,k][1] = 0.0
        vel[i,j,k][2] = 0.0
        for q in ti.static(range(15)):
            f_old[i,j,k][q] = f_new[i,j,k][q]
            rho[i,j,k] += f_new[i,j,k][q]
            vel[i,j,k][0] += (ti.cast(e[k,0],ti.float)*f_new[i,j,k][q])
            vel[i,j,k][1] += (ti.cast(e[k,1],ti.float)*f_new[i,j,k][q])
            vel[i,j,k][2] += (ti.cast(e[k,2],ti.float)*f_new[i,j,k][q])

        vel[i,j,k][0] = vel[i,j,k][0] / rho[i,j,k]
        vel[i,j,k][1] = vel[i,j,k][1] / rho[i,j,k]
        vel[i,j,k][2] = vel[i,j,k][2] / rho[i,j,k]

def write_fluid_vtk(istep,params,vel):
    #write out fluid velocity and fluid vorticity to legacy VTK 2.0 files
    filename_vel = "vel"+str(istep).rjust(8,"0")+".vtk"
    filename_vor = "vor"+str(istep).rjust(8,"0")+".vtk"
    f_vel = open(filename_vel,"w+")
    f_vel.write("# vtk DataFile Version 2.0")
    f_vel.write(filename)
    f_vel.write('ASCII')
    f_vel.write('DATASET STRUCTURED_POINTS')
    f_vel.write('DIMENSIONS '+str(params.nx)+" "+str(params.ny)+" "+str(params.nz))
    f_vel.write('ORIGIN 0 0 0')
    f_vel.write('SPACING 1 1 1')
    f_vel.write('POINT_DATA '+str(params.nx*params.ny))
    f_vel.write('VECTORS velocity_field float')
    for k in range(vel.shape[2]):
        for j in range(vel.shape[1]):
            for i in range(vel.shape[0]):
                f_vel.write(str(vel[i,j,k,0])+" "+str(vel[i,j,k,1]+" "+str(vel[i,j,k,2])))
    f_vel.close()

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
def calculate_stress():
    dx = ti.Vector([1,0,0])
    dy = ti.Vector([0,1,0])
    dz = ti.Vector([0,0,1])
    for x,y,z in Q:
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                P0 = 0.0 # TODO need to add hydrostatic pressure from distortion free energy
                s1 = -1.0*P0*kdelta[i,j]
                
                # s2 term, 2*eps*(Qij + 1/3 deltaij)*Qkl*Hkl
                sumQH = 0.0
                for iter_k in ti.static(range(3)):
                    for iter_l in ti.static(range(3)):
                        sumQH += Q[x,y,z][iter_k,iter_l]*H[x,y,z][iter_k,iter_l]
                s2 = 2*eps*(Q[x,y,z][i,j] + kdelta[i,j]/3)*sumQH

                #s3 term, eps*Hij*(Qkj + 1/3 deltakj)
                sumQ = 0.0
                for iter_k in ti.static(range(3)):
                    deltakj = 0.0
                    if iter_k == j:
                        deltakj = 1.0
                    sumQ = Q[x,y,z][iter_k,j] + deltakj/3.0
                s3 = -1.0*eps*H[x,y,z][i,j] * sumQ  

                #s4 term, -eps*(Qik + 1/3 deltaik)*Hkj
                s4 = 0.0
                for iter_k in ti.static(range(3)):
                    deltaik = 0.0
                    if iter_k == i:
                        deltaik = 1.0
                    s4 += H[x,y,z][iter_k,j]*(Q[x,y,z][i,iter_k] + deltaik/3)
                s4 = s4 * (-1.0*eps)

                #s5 term = d_i Qkm * d_j Qkm i and j correspond to i,j of stress tensor sigma_ij
                s5 = 0.0
                for iter_gamma in ti.static(range(3)):
                    for iter_nu in ti.static(range(3)):
                        xpi = x+dx[i]
                        xpj = x+dx[j]
                        if xpi >= lx-1:
                            xpi = 0
                        if xpj >= lx-1:
                            xpj = 0
                        ypi = y+dy[i]
                        ypj = y+dy[j]
                        if ypi >= ly-1:
                            ypi = 0
                        if ypj >= ly-1:
                            ypj = 0
                        zpi = z+dz[i]
                        zpj = z+dz[i]
                        if zpi >= lz-1:
                            zpi = 0
                        if zpj >= lz-1:
                            zpj = 0
                        s5 += (Q[x+dx[j],y+dy[j],z+dz[j]][iter_gamma,iter_nu] - Q[x,y,z][iter_gamma,iter_nu]) * K * (Q[x+dx[i],y+dy[i],z+dz[i]][iter_gamma,iter_nu] - Q[x,y,z][iter_gamma,iter_nu])

                #s6 term
                s6 = 0.0
                for iter_k in ti.static(range(3)):
                    s6 += Q[x,y,z][i,iter_k]*H[x,y,z][iter_k,j]

                s7 = 0.0
                for iter_k in ti.static(range(3)):
                    s7 += H[x,y,z][i,iter_k]*Q[x,y,z][iter_k,j]

                stress_sym[x,y,z][i,j] = s1+s2+s3+s4+s5
                stress_anti[x,y,z][i,j] = s6+s7
                stress[x,y,z][i,j] = s1 + s2 + s3 + s4 + s5 + s6 + s7
              
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
    init_lb()
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