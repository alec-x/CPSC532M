using Printf
include("misc.jl")
include("findMin.jl")

# Fits a logistic regression model with L2-regularization
function logRegL1(X,y,lambda)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticL1Obj(w,X,y,lambda)

	# Solve least squares problem
	w = findMinL1(funObj,w, lambda)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticL1Obj(w,X,y,lambda)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw))) + lambda*sum(abs.(w))
	g = (-X'*(y./(1 .+ exp.(yXw))))
	return (f,g)
end
