from src import qij_solvers
from src import qij_io

name = "disc_line"
nsteps = 150
dims = (0,0,0,1.2,1.2,0.2)
delta = 0.02
dt = 1e-3
isave = (True,1) # saves every step
S0 = 0.53 # order parameter
A = -1/100
B = -12.3/100
C = 10/100
L = 2.32/100
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
bc = {'top':{'defect':(1,-0.5)},
      'bot':{'defect':(1,0.5)},
      'left':{'anch':(0,1,0)},
      'right':{'anch':(1,0,0)},
      'front':{'anch':(0,1,0)},
      'back':{'anch':(0,1,0)}}

qij_io.setup(name,isave)

sim = qij_solvers.RelaxationQij3DTensor(name,dims,delta,nsteps,dt,isave)
sim.initialize(A,B,C,L,S0)
sim.run(boundary_conditions=bc)