using Printf
include("misc.jl")
include("findMin.jl")

using JLD
data = load("multiData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

softMaxClassifier(X,y)

function softMaxClassifier(X,y)

	(n,d) = size(X)
	k = size(unique(y))[1]
	# Initial guess
	w = zeros(500)

	funObj(w) = softMaxObj(w,X,y)
	w = findMin(funObj,w,verbose=false, derivativeCheck=true)
end

function softMaxObj(w,X,y)
	(n,d) = size(X)
	k = trunc(Int,size(w)[1]/d) # number of classes
	wNew = reshape(w, k, d)
	f = 0
	g = 0
	gradMatrix = zeros(k,d)

	for i in 1:n
		global f += ((-1)*(X*wNew')[i,y[i]] + log(sum(exp.((X*wNew')[i,:]))))[1]
	end

	# Calculates the kxd matrix of partial derivatives
	for c in 1:k
		for j in 1:d
			for i in 1:n
				global g += ((-1)*(y .== c)[i] + ((1 / sum(exp.((X*wNew')[i,:]))) * (exp((X*wNew')[i,c])))) * X[i,j]
			end
			gradMatrix[c,j] = g
		end
	end

	g = reshape(gradMatrix, d*k, 1) # Reshape g from kxd to dkx1 for use in findMin

	return(f,g)
end
