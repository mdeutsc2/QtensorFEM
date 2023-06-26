from src import qij_solvers
from src import qij_io

name = "defect_arc3"
nsteps = 75
dims = (0,0,0,1.2,1.2,0.1)
delta = 0.02
dt = 1e-3
isave = (True,1) # saves every step
S0 = 0.53 # order parameter
A = -1
B = -12.3
C = 10
L = 2.32
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
bc = {'top':{'twist':((1,0,0),0.9)}, # initial anchoring, twist/step in deg
      'bot':{'defect':(2,0.16)}} # number of defects, spacing
# bc = {'top':{'anch':(1,0,0)},
#       'bot':{'defect':(2,0.16)}}

qij_io.setup(name,isave)

sim = qij_solvers.RelaxationQij3DTensor(name,dims,delta,nsteps,dt,isave)
sim.initialize(A,B,C,L,S0)
sim.run(boundary_conditions=bc)