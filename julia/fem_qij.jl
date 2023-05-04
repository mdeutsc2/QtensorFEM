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
dt = 0.01
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

dQ = OffsetArray{Float64}(undef,1:length(mesh.nodes),0:4)
volumes = zeros(length(mesh.cells),4,1)
M_inv = Array{Float64}(undef,length(mesh.cells),4,4)
# accessing all the nodes and their positions
for i in 1:2#getncells(mesh)
    icell = getcells(mesh,i)
    println(i,"\t",icell)
    temp = zeros(4,4)
    for (index,inode) in enumerate(icell.nodes)
        println(index,"\t",inode,"\t",getnodes(mesh,inode).x)
        node_pos = getnodes(mesh,inode).x
        temp[index,1] = 1.0
        temp[index,2] = node_pos[1]
        temp[index,3] = node_pos[2]
        temp[index,4] = node_pos[3]
        #TODO calculate volumes associated with each node
    end
    #matrix inversion
    println(temp)
    println(inv(temp))
    println(temp.*inv(temp))
    println(temp/Matrix(1.0I, 4, 4))
    M_inv[i,:,:] .= inv(temp)
    #println(inv(temp).*temp)
    #@assert((inv(temp).*temp .- one(SymmetricTensor{2, 4})) <= 1.0e-5)
end


# while t < T
#    t += dt

# calculate distortion term

# calculate bulk term

# update Q
# Q = Q_new
println("Done!")