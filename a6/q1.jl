using Printf
using Statistics
using LinearAlgebra
include("misc.jl")
include("findMin.jl")

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repeat(mu,t,1)
end

function robustPCA(X,k, epsilon)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X
    f = huberLoss(R, epsilon)
    funObjZ(z) = pcaObjZ(z,X,W, epsilon)
    funObjW(w) = pcaObjW(w,X,Z, epsilon)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjW,W[:],verbose=false,maxIter=10)

        R = Z*W - X
        f = huberLoss(R, epsilon)
        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescent(Xhat,W,mu, epsilon)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compress_gradientDescent(Xhat,W,mu, epsilon)
    (t,d) = size(Xhat)
    k = size(W,1)
    Xcentered = Xhat - repeat(mu,t,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZ(z,Xcentered,W, epsilon)
    Z[:] = findMin(funObj,Z[:],verbose=false)
    return Z
end


function pcaObjZ(z,X,W, epsilon)
    # Rezie vector of parameters into matrix
    n = size(X,1)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X
    f = huberLoss(R, epsilon)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by W' to get elements of gradient
	case = abs.(dR) .<= epsilon
	close = findall(case)
	far = findall(.!case)
	dR[far] = epsilon * sign.(dR[far])

	G = dR*W'

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjW(w,X,Z, epsilon)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X
    f = huberLoss(R, epsilon)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by Z' to get elements of gradient
	case = abs.(dR) .<= epsilon
	close = findall(case)
	far = findall(.!case)
	dR[far] = epsilon * sign.(dR[far])

	G = Z'dR

    # Return function and gradient vector
    return (f,G[:])
end

function huberLoss(R, epsilon)
	case = abs.(R) .<= epsilon
	close = findall(case)
	far = findall(.!case)
	f = sum((1/2)R[close].^2) + epsilon*sum(abs.(R[far]) .- (1/2)epsilon)
    return f
end
