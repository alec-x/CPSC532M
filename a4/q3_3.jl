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
	w = zeros(k*d)

	funObj(w) = softMaxObj(w,X,y)
	w = findMin(funObj,w,verbose=false, derivativeCheck=true)
end

function softMaxObj(w,X,y)
	(n,d) = size(X)
	k = trunc(Int,size(w)[1]/d) # number of classes
	wNew = reshape(w, k, d)

	gradMatrix = zeros(k,d)

	for c in 1:k
		for j in 1:d
			for i in 1:n
				g +=  ((y .== c)[i] + (1 / sum(exp.((X*wNew')[i,:]))) * (exp((X*wNew')[i,c]))) * X[i,j]
			end
			gradMatrix[c,j] = g
		end
	end

	return(f,g)
end
