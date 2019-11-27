include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X, k)
    (n,d) = size(X)

    # Compute all distances
    D = ISOMAP(X,k)

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end

function ISOMAP(X, k)
    (n,d) = size(X)
    k = min(n,k) # To save you some debuggin
	G = zeros(n,n)

    for i in 1 : n
	  distances = distancesSquared(X,X[i,:]')
      sortedDist = sortperm(distances[:,1])[2:k+1] # dont include distance to self

	  for j in 1 : size(sortedDist)[1]
	  	G[i,sortedDist[j]] = distances[sortedDist[j]]
  	  end
    end
	D = zeros(n,n)
	for i in 1:n
	  for j in 1:n
		D[i,j] = dijkstra(G,i,j)
	  end
	end
    return G
end
