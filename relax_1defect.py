from src import qij_solvers

name = "relax_1defect"
nsteps = 50
dims = (0,0,0,2.5,2.5,0.5)
delta = 0.05
dt = 1e-3
isave = (True,1) # saves every step
S0 = 0.53 # order parameter
A = -1
B = -12.3
C = 10
L = 2.32
theta = 1 #time-step family, theta=1 -> backwards Euler, theta=0.5 -> Crank-Nicholson, theta = 0 -> forwards Euler
bc = {'top':{'defect':1},
      'bot':{'defect':1}}


sim = qij_solvers.RelaxationQij3DTensor(name,dims,0.05,nsteps,dt,isave)
print(type(sim))
sim.initialize(A,B,C,L,S0)
sim.run(boundary_conditions=bc)