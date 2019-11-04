using Printf
using Statistics
include("misc.jl")
include("findMin.jl")

using JLD
data = load("multiData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit one-vs-all logistic regression model
include("softMax.jl")
model = softMaxClassifier(X,y)

# Compute training and validation error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@show(trainError)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)
@show(validError)

# Plot results
#=
k = maximum(y)
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model,Xtest=Xtest,ytest=ytest,biasIncluded=true,k=5)
=#
