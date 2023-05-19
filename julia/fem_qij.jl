using Ferrite
using SparseArrays
using Tensors
using OffsetArrays
using LinearAlgebra



function setup_defects(defects,Lx,Ly,Lz,nx,ny,nz)
    println("Defects:")
    ndefects = defects["ndefects"]
    if length(size(ndefects)) == 1
        ydefects = 1
        xdefects = size(ndefects)[1]
    else
        ydefects = size(ndefects)[2]
        xdefects = size(ndefects)[1]
    end
    #spacing_string = split(defects["spacing"]),",")
    print(defects["spacing"])
    spacing = defects["spacing"]
    #spacing = (parse(Int,defects["spacing"][1][2:end]),parse(Int,spacing[2][1]))
    th = zeros(Lx,Ly)
    xx = zeros((xdefects,ydefects))
    yy = zeros((xdefects,ydefects))
    q = zeros((xdefects,ydefects))
    
    if sum(size(ndefects)) > 1
        id = 0
        for ii in 1:xdefects
            for jj in 1:ydefects
                id += 1
                if !isempty(spacing) # test if spacing parameters are specified
                    if spacing[1] != 0
                        xx[id] = (ii*spacing[1] + 0.5) + (1-(spacing[1]*(xdefects+1))/Lx)*(Lx/2)
                    else
                        xx[id] = ii*(Lx/(xdefects+1))+0.5
                    end
                    if spacing[2] != 0 
                        yy[id] = (jj*spacing[2] + 0.5) + (1-(spacing[2](ydefects+1))/Ly)*(Ly/2)
                    else
                        yy[id] = jj*(Ly/(ydefects+1))+0.5
                    end
                else 
                    xx[id] = ii*(Lx/(xdefects+1))+0.5
                    yy[id] = jj*(Ly/(ydefects+1))+0.5
                end
                q[id] = ndefects[ii,jj]
                println("\t",xx[id],"\t",yy[id],"\t",q[id])
            end
        end

    else
        id = 1
        xx[1] = Lx/2 +0.5
        yy[1] = Ly/2 +0.5
        q[1] = 0.5
    end
    """q = [0.5,-0.5,0.5,-0.5,
        1.0,-1.0,1.0,-1.0,
        -1.0,1.0,-1.0,1.0,
        -0.5,0.5,-0.5,0.5]"""

    for idefect = 1:id
        for i = 1:Lx
            for j = 1:Ly
                phi = atan(j-yy[idefect],i-xx[idefect])
                th[i,j] += q[idefect]*phi
            end
        end
    end
    k = 1 #for the bottom substrate
    for i = 1:Lx
        for j = 1:Ly
            nx[i,j,k] = 0.5*cos(th[i,j]+pi/4.0)
            ny[i,j,k] = 0.5*sin(th[i,j]+pi/4.0)
            nz[i,j,k] = 0
        end
    end
    return nx,ny,nz
end

# PARAMETERS
Lx = 20
Ly = 20
Lz = 20
S0 = 0.53
A = 1.0
B = 1.0
C = 1.0
K = 1.0
Gamma = 1 # rotational viscosity
dt = 1.0e-6
T = 1.0
nsteps = T/dt

defects = Dict("ndefects" => [0.5,-0.5], "spacing" => (8,0))

mesh = generate_grid(Tetrahedron, (Lx,Ly,Lz))

Lx_max = maximum([inode.x[1] for inode in mesh.nodes])
Ly_max = maximum([inode.x[2] for inode in mesh.nodes])
Lz_max = maximum([inode.x[3] for inode in mesh.nodes])

Lx_min = minimum([inode.x[1] for inode in mesh.nodes])
Ly_min = minimum([inode.x[2] for inode in mesh.nodes])
Lz_min = minimum([inode.x[3] for inode in mesh.nodes])

dx = (Lx_max-Lx_min)/(Lx)
dy = (Ly_max-Ly_min)/(Ly)
dz = (Lz_max-Lz_min)/(Lz)

println("Domain:\n X:",Lx_min," -> ",Lx_max,"\t",dx)
println(" Y:",Ly_min," -> ",Ly_max,"\t",dy)
println(" Z:",Lz_min," -> ",Lz_max,"\t",dz)


println("Num cells:",length(mesh.cells))
println("Num nodes:",length(mesh.nodes))

# defining a q tensor for each node
Q_full = zeros(length(mesh.nodes),3,3)
#nx,ny,nz = setup_defects(defects,Lx,Ly,Lz,zeros(Lx+1,Ly+1,Lz+1),zeros(Lx+1,Ly+1,Lz+1),zeros(Lx+1,Ly+1,Lz+1))
nx = rand(Float64,(Lx+1,Ly+1,Lz+1))
ny = rand(Float64,(Lx+1,Ly+1,Lz+1))
nz = rand(Float64,(Lx+1,Ly+1,Lz+1))
s = nx.*nx + ny.*ny + nz.*nz
nx = nx./s
ny = ny./s
nz = nz./s

nx = reshape(nx,(length(mesh.nodes))) 
ny = reshape(ny,(length(mesh.nodes)))
nz = reshape(nz,(length(mesh.nodes)))
@assert(length(nx) == length(mesh.nodes))
# assign values of q to each node
for inode in 1:length(mesh.nodes)
    #println(grid.nodes[inode])
    n = zeros(3)
    n[1] = nx[inode]
    n[2] = ny[inode]
    n[3] = nz[inode]
    for a in 1:3
        for b in 1:3
            Q_full[inode,a,b] = S0*(n[a]*n[b] - (I[a,b]*1.0)/3)
        end
    end
end

# q0 q2 q3
# q2 q1 q4
# q3 q4 -q1-q0
Q = OffsetArray{Float64}(undef,1:length(mesh.nodes),0:4)
Q[:,0] .= Q_full[:,1,1]
Q[:,1] .= Q_full[:,2,2]
Q[:,2] .= Q_full[:,1,2]
Q[:,3] .= Q_full[:,1,3]
Q[:,4] .= Q_full[:,2,3]
Q_new = deepcopy(Q)

println("Q: ",size(Q))

dQ = OffsetArray{Float64}(0.0,1:length(mesh.nodes),0:4)
volumes = zeros(length(mesh.nodes))
M_inv = Array{Float64}(undef,length(mesh.cells),4,4)
# accessing all the nodes and their positions
for i in 1:getncells(mesh)
    icell = getcells(mesh,i)
    #println(i,"\t",icell)
    # calcuating the volume for each cell
    ab = [getnodes(mesh,icell.nodes[2]).x[1] - getnodes(mesh,icell.nodes[1]).x[1],
          getnodes(mesh,icell.nodes[2]).x[2] - getnodes(mesh,icell.nodes[1]).x[2],
          getnodes(mesh,icell.nodes[2]).x[3] - getnodes(mesh,icell.nodes[1]).x[3]]
    ac = [getnodes(mesh,icell.nodes[3]).x[1] - getnodes(mesh,icell.nodes[1]).x[1],
          getnodes(mesh,icell.nodes[3]).x[2] - getnodes(mesh,icell.nodes[1]).x[2],
          getnodes(mesh,icell.nodes[3]).x[3] - getnodes(mesh,icell.nodes[1]).x[3]]
    ad = [getnodes(mesh,icell.nodes[4]).x[1] - getnodes(mesh,icell.nodes[1]).x[1],
          getnodes(mesh,icell.nodes[4]).x[2] - getnodes(mesh,icell.nodes[1]).x[2],
          getnodes(mesh,icell.nodes[4]).x[3] - getnodes(mesh,icell.nodes[1]).x[3]]
    icell_volume = dot(cross(ab,ac),ad)/6
    inode_volume = 0.25*icell_volume
    temp = zeros(4,4)
    for (index,inode) in enumerate(icell.nodes)
        #println(index,"\t",inode,"\t",getnodes(mesh,inode).x)
        node_pos = getnodes(mesh,inode).x
        temp[index,1] = 1.0
        temp[index,2] = node_pos[1]
        temp[index,3] = node_pos[2]
        temp[index,4] = node_pos[3]
        volumes[inode] = inode_volume
        #println(temp[index,:])
    end
    M_inv[i,:,:] .= inv(temp)
    #println(inv(temp).*temp)
    #@assert((inv(temp).*temp .- one(SymmetricTensor{2, 4})) <= 1.0e-5)
end


t = 0.0
E_dist = 0.0
while t < T
# calculate distortion term
    for i in 1:getncells(mesh)
        icell = getcells(mesh,i)
        M = M_inv[i,:,:]     
        vol = volumes[icell.inode[0]]
        #Expansion coefficients for q0:
        #a2=dq0/dx a3 = dq0/dy a4=dq0/dz
        #a2=qq0(1)*m(2,1)+qq0(2)*m(2,2)+qq0(3)*m(2,3)+qq0(4)*m(2,4) 
        a2=Q[icell.nodes[1],0]*M[2,1] + Q[icell.nodes[2],0]*M[2,2] + Q[icell.nodes[3],0]*M[2,3] + Q[icell.nodes[4],0]*M[2,4]
        #a3=qq0(1)*m(3,1)+qq0(2)*m(3,2)+qq0(3)*m(3,3)+qq0(4)*m(3,4)
        a3=Q[icell.nodes[1],0]*M[3,1] + Q[icell.nodes[2],0]*M[3,2] + Q[icell.nodes[3],0]*M[3,3] + Q[icell.nodes[4],0]*M[3,4]
        #a4=qq0(1)*m(4,1)+qq0(2)*m(4,2)+qq0(3)*m(4,3)+qq0(4)*m(4,4)
        a4=Q[icell.nodes[1],0]*M[4,1] + Q[icell.nodes[2],0]*M[4,2] + Q[icell.nodes[3],0]*M[4,3] + Q[icell.nodes[4],0]*M[4,4]
        
        #Expansion coefficients for q1:
        #b2=dq1/dx b3 = dq1/dy b4=dq1/dz
        #b2=qq1(1)*m(2,1)+qq1(2)*m(2,2)+qq1(3)*m(2,3)+qq1(4)*m(2,4)
        #b3=qq1(1)*m(3,1)+qq1(2)*m(3,2)+qq1(3)*m(3,3)+qq1(4)*m(3,4)
        #b4=qq1(1)*m(4,1)+qq1(2)*m(4,2)+qq1(3)*m(4,3)+qq1(4)*m(4,4)
        b2=Q[icell.nodes[1],1]*M[2,1] + Q[icell.nodes[2],1]*M[2,2] + Q[icell.nodes[3],1]*M[2,3] + Q[icell.nodes[4],1]*M[2,4]
        b3=Q[icell.nodes[1],1]*M[3,1] + Q[icell.nodes[2],1]*M[3,2] + Q[icell.nodes[3],1]*M[3,3] + Q[icell.nodes[4],1]*M[3,4]
        b4=Q[icell.nodes[1],1]*M[4,1] + Q[icell.nodes[2],1]*M[4,2] + Q[icell.nodes[3],1]*M[4,3] + Q[icell.nodes[4],1]*M[4,4]

        #Expansion coefficients for q2:
        #c2=dq2/dx c3 = dq2/dy c4=dq2/dz
        #c2=qq2(1)*m(2,1)+qq2(2)*m(2,2)+qq2(3)*m(2,3)+qq2(4)*m(2,4) 
        #c3=qq2(1)*m(3,1)+qq2(2)*m(3,2)+qq2(3)*m(3,3)+qq2(4)*m(3,4) 
        #c4=qq2(1)*m(4,1)+qq2(2)*m(4,2)+qq2(3)*m(4,3)+qq2(4)*m(4,4)
        c2=Q[icell.nodes[1],2]*M[2,1] + Q[icell.nodes[2],2]*M[2,2] + Q[icell.nodes[3],2]*M[2,3] + Q[icell.nodes[4],2]*M[2,4]
        c3=Q[icell.nodes[1],2]*M[3,1] + Q[icell.nodes[2],2]*M[3,2] + Q[icell.nodes[3],2]*M[3,3] + Q[icell.nodes[4],2]*M[3,4]
        c4=Q[icell.nodes[1],2]*M[4,1] + Q[icell.nodes[2],2]*M[4,2] + Q[icell.nodes[3],2]*M[4,3] + Q[icell.nodes[4],2]*M[4,4]


        #Expansion coefficients for q3:
        #d2=dq3/dx d3 = dq3/dy d4=dq3/dz
        # d2=qq3(1)*m(2,1)+qq3(2)*m(2,2)+qq3(3)*m(2,3)+qq4(4)*m(2,4) 
        # d3=qq3(1)*m(3,1)+qq3(2)*m(3,2)+qq3(3)*m(3,3)+qq4(4)*m(3,4) 
        # d4=qq3(1)*m(4,1)+qq3(2)*m(4,2)+qq3(3)*m(4,3)+qq4(4)*m(4,4)
        d2=Q[icell.nodes[1],3]*M[2,1] + Q[icell.nodes[2],3]*M[2,2] + Q[icell.nodes[3],3]*M[2,3] + Q[icell.nodes[4],3]*M[2,4]
        d3=Q[icell.nodes[1],3]*M[3,1] + Q[icell.nodes[2],3]*M[3,2] + Q[icell.nodes[3],3]*M[3,3] + Q[icell.nodes[4],3]*M[3,4]
        d4=Q[icell.nodes[1],3]*M[4,1] + Q[icell.nodes[2],3]*M[4,2] + Q[icell.nodes[3],3]*M[4,3] + Q[icell.nodes[4],3]*M[4,4]

        
        #Expansion coefficients for q4:
        #e2=dq4/dx e3 = dq4/dy e4=dq4/dz
        # e2=qq4(1)*m(2,1)+qq4(2)*m(2,2)+qq4(3)*m(2,3)+qq4(4)*m(2,4) 
        # e3=qq4(1)*m(3,1)+qq4(2)*m(3,2)+qq4(3)*m(3,3)+qq4(4)*m(3,4) 
        # e4=qq4(1)*m(4,1)+qq4(2)*m(4,2)+qq4(3)*m(4,3)+qq4(4)*m(4,4)
        e2=Q[icell.nodes[1],4]*M[2,1] + Q[icell.nodes[2],4]*M[2,2] + Q[icell.nodes[3],4]*M[2,3] + Q[icell.nodes[4],4]*M[2,4]
        e3=Q[icell.nodes[1],4]*M[3,1] + Q[icell.nodes[2],4]*M[3,2] + Q[icell.nodes[3],4]*M[3,3] + Q[icell.nodes[4],4]*M[3,4]
        e4=Q[icell.nodes[1],4]*M[4,1] + Q[icell.nodes[2],4]*M[4,2] + Q[icell.nodes[3],4]*M[4,3] + Q[icell.nodes[4],4]*M[4,4]
        
        E_dist_local= volumes[icell.node[1]]*((a2*a2)+(a3*a3)+(a4*a4)+
                                              (b2*b2)+(b3*b3)+(b4*b4)+
                                              (c2*c2)+(c3*c3)+(c4*c4)+
                                              (d2*d2)+(d3*d3)+(d4*d4)+
                                              (e2*e2)+(e3*e3)+(e4*e4)+
                                              (a2*b2)+(a3*b3)+(a4*b4))
        E_dist += E_dist_local
        ddQ = OffsetArray{Float64}(0.0,1:4,0:4)
        for (i,inode) in enumerate(icell.nodes)
            ddQ[i,0] -= vol*((2*a2*M[2,i])+(2*a3*M[3,i])+(2*a4*M[4,i])+(b2*M[2,i])+(b3*M[3,i])+(b4*M[4,i]))
            ddQ[i,1] -= vol*((2*b2*M[2,i]))+(2*b3*M[3,i])+(2*b4*M[4,i] +(a2*M[2,i])+(a3*M[3,i])+(a4*M[4,i]))
            ddQ[i,2] -= 2*vol*((c2*M[2,1])+(c3*M[3,1])+(c4*M[4,1])) 
            ddQ[i,3] -= 2*vol*((d2*M[2,1])+(d3*M[3,1])+(d4*M[4,1])) 
            ddQ[i,4] -= 2*vol*((e2*M[2,1])+(e3*M[3,1])+(e4*M[4,1]))
        end
        for (i,inode) in enumerate(icell.nodes)
            dQ[inode,0] += ddQ[i,0]
            dQ[inode,1] += ddQ[i,1]
            dQ[inode,2] += ddQ[i,2]
            dQ[inode,3] += ddQ[i,3]
            dQ[inode,4] += ddQ[i,4]
        end
    end
# calculate bulk term
    #for inode in 1:length(mesh.nodes)
# update Q
    Q = Q_new
    global t += dt
end

println("Done!")