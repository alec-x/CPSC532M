using Printf
include("misc.jl")
include("findMin.jl")

# Variant where we use forward selection for feature selection
function logRegL0(X,y,lambda)
	(n,d) = size(X)

	# Define an objective that will operate on a subset of the data called Xs
	funObj(w) = logisticObj(w,Xs,y)

	# Start out just using the bias variable (assumed to be in first column),
	# and record 'score' which is the loss plus regularizer
	S = [1] # Candidate set of features
	Xs = X[:,S]
	w = zeros(length(S),1)
	w = findMin(funObj,w,verbose=false)
	(f,~) = funObj(w)
	score = f + lambda*length(S)
	minScore = score # Lowest score we've found
	minS = S # Best set of features we've found

	@show(minScore)
	@show(minS)

	# Greedily start adding the variable that improves the score the most
	oldScore = Inf
	while minScore != oldScore
		oldScore = minScore

		# Print out the variables we've selected so far
		@printf("Current set of selected variables (score = %f):\n",minScore)
		for j in 1:length(S)
			@printf("%d ",S[j])
		end
		@printf("\n")

		for j in setdiff(1:d,S)
			# Fit the model with 'j' added to the feature set 'S'
			# then compute the score and update 'minScore' and 'minS'
			Sj = [S;j]
			Xs = X[:,Sj]

			funObj(w) = logisticObjL0(w, X, y, lambda)
		end
		S = minS
	end

	# Construct final 'w' vector
	w = zeros(d,1)
	S = minS
	Xs = X[:,S]
	w[S] = findMin(funObj,zeros(length(S),1),verbose=false)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObjL0(w,X,y, lambda)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end
