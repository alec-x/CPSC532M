using LinearAlgebra
include("misc.jl")

function leastSquaresKernelBasis(X,y,lambda,p)
	n = size(X,1)
    u = (polyKernel(X,X,p) + lambda*I)^-1*y
	# Make linear prediction function
	predict(Xhat) = polyKernel(X, Xhat,p)*u

	# Return model
	return LinearModel(predict, u)
end

function polyKernel(X1,X2, p)
	# x1_i * x2_j, should be 1xn * nx1 for each element
	n = size(X,1)
	K = zeros(n,n)
	for i in 1:n
		for j in 1:n
			K[i,j] = (1 + X1[i,:]'*X2[j,:])^p
		end
	end

	return K
end
