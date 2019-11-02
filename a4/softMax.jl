function softMaxClassifier(X,y)
	funObj(w) = softMaxObj(w,X,y)
	(n,d) = size(X)
	k = size(unique(y))[1]
	# Initial guess
	w = zeros(d*k) .+ 10
	w = findMin(funObj,w,verbose=false, derivativeCheck=true)
	w = reshape(w, k, d)
	predict(Xhat) = sort!(Xhat*w',dims=2)[:,end]

	return LinearModel(predict,w)
end

function softMaxObj(w,X,y)
	(n,d) = size(X)
	k = trunc(Int,size(w)[1]/d) # number of classes
	wNew = reshape(w, k, d)
	f = 0
	g = 0
	gradMatrix = zeros(k,d)

	for i in 1:n
		f += -(wNew[y[i],:]'*X[i,:]) + log(sum(exp.((wNew*X[i,:]))))
	end

	# Calculates the kxd matrix of partial derivatives
	for c in 1:k
		for j in 1:d
			gradMatrix[c,j] = sum((sum(exp.(wNew*X'), dims=1).^-1 .* exp.(wNew[c,:]'*X') .+ -(y .== c)').*X[:,j]')
		end
	end

    gradMatrix = reshape(gradMatrix, d*k, 1) # Reshape g from kxd to dkx1 for use in findMin

	return (f,gradMatrix)
end
