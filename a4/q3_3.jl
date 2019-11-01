using Printf
include("misc.jl")
include("findMin.jl")

using JLD
data = load("multiData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

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

	for c in 1:k
		for j in 1:ds
		wNew[c,j] = (y .== c).*(-X[:,c]) + #...
	end
end
