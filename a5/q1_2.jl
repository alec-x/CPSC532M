using LinearAlgebra
include("misc.jl")

function leastSquaresKernelBasis(X,y,lambda,p)
	n = size(X,1)
    u = (polyKernel(X,X,p) + lambda*I)^-1*y
	# Make linear prediction function
	predict(Xhat) = polyKernel(Xhat, X,p)*u

	# Return model
	return LinearModel(predict, u)
end

function polyKernel(X1,X2, p)
	# x1_i * x2_j, should be 1xn * nx1 for each element
	n1 = size(X1,1)
	n2 = size(X2,1)

	K = zeros(n1,n2)
	for i in 1:n1
		for j in 1:n2
			K[i,j] = (1 + X1[i,:]'*X2[j,:])^p
		end
	end

	return K
end
