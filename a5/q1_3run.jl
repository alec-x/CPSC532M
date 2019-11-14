using Printf
using Statistics

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("q1_3.jl")

lambda = 1e-3
sigma = 1

model = GaussianRBFKernelBasis(X,y, lambda, sigma)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)

# Plot model
using PyPlot
figure(1)
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
yhat = model.predict(Xhat[:,:])
plot(Xhat,yhat,"g")
show()
