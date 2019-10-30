using Printf
include("misc.jl")
include("findMin.jl")

# Fits a logistic regression model with L2-regularization
function logRegL2(X,y,lambda)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticL2Obj(w,X,y,lambda)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticL2Obj(w,X,y,lambda)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw))) + ((lambda/2)*(w'*w))[1]
	g = (-X'*(y./(1 .+ exp.(yXw)))) + (lambda * w)
	return (f,g)
end
