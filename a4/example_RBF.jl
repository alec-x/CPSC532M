using Printf
using Random

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)
validStart = Int64(n/2+1) # Start of validation indices
validEnd = Int64(n) # End of validation incides
validNdx = perm[validStart:validEnd] # Indices of validation examples
trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples
Xtrain = X[trainNdx,:]
ytrain = y[trainNdx]
Xvalid = X[validNdx,:]
yvalid = y[validNdx]

# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
minErr = Inf
bestSigma = []
lambda = 1e-12
for sigma in 2.0.^(-15:15)
	# Train on the training set
	model = leastSquaresRBF(Xtrain,ytrain,sigma, lambda)

	# Compute the error on the validation set
	yhat = model.predict(Xvalid)
	validError = sum((yhat - yvalid).^2)/(n/2)
	@printf("With sigma = %.3f, , lambda = %.3f, validError = %.2f\n",sigma, lambda, validError)

	# Keep track of the lowest validation error
	if validError < minErr
		global minErr = validError
		global bestSigma = sigma
	end

end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
