# initial condition and boundary condition interpolation functions
import numpy as np

def initQ3d_rand(x,S0):
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

def initQ3d_anch(x,S0):
    values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
    n = np.zeros((3,x.shape[1])) # director
    n[0,:] = 1.0
    n[1,:] = 0.0
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

def initQ2d_rand(x):
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

def initQ3d_defects(x):
    values = np.zeros((3*3,x.shape[1]),dtype=np.float64)
    n = np.zeros((3,x.shape[1]))
    theta = np.zeros((axispts,axispts))
    w = 0.25 # defect spacing
    theta = 0.5*np.arctan(x[1]/(x[0]+w/2))-0.5*np.arctan(x[1]/(x[0]-w/2)) + 0.5*np.pi
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
    print(theta.shape)
    return values