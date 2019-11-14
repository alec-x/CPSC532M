using LinearAlgebra
include("misc.jl")

function GaussianRBFKernelBasis(X,y,lambda,sigma)
	n = size(X,1)
    u = (GaussianRBFKernel(X,X,sigma) + lambda*I)^-1*y
	# Make linear prediction function
	predict(Xhat) = GaussianRBFKernel(Xhat, X, sigma)*u

	# Return model
	return LinearModel(predict, u)
end

function GaussianRBFKernel(X1,X2, sigma)
	# x1_i * x2_j, should be 1xn * nx1 for each element
	n1 = size(X1,1)
	n2 = size(X2,1)

	K = zeros(n1,n2)

	for i in 1:n1
		for j in 1:n2
			K[i,j] = exp(-distancesSquared(X1[i,:]', X2[j,:]')/(2*sigma^2 ))[1]

		end
	end

	return K
end
