using Printf
using Random

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is sorted, so *randomly* split into train and validation:
(n,d) = size(X)
k = 10
perm = randperm(n)


global validEnd = 0
Xtrain = zeros(Int64(n*(k-1)/k),d,k)
ytrain = zeros(Int64(n*(k-1)/k),d,k)
Xvalid = zeros(Int64(n/k),d,k)
yvalid = zeros(Int64(n/k),d,k)
for i in 1:k
	validStart = validEnd + 1 # Start of validation indices
	global validEnd = Int64(i*n/k) # End of validation incides

	validNdx = perm[validStart:validEnd] # Indices of validation examples
	trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples

	Xtrain[:,:,i] = X[trainNdx,:]
	ytrain[:,:,i] = y[trainNdx]
	Xvalid[:,:,i] = X[validNdx,:]
	yvalid[:,:,i] = y[validNdx]
end

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
