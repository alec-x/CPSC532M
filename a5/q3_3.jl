using DelimitedFiles
using PyPlot

# Load data
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)

include("PCA.jl")
k = 5
model = PCA(X,k)

Z = model.compress(X)
Xpred = model.expand(Z)
figure(1)
clf()
plot(Z[:,1],Z[:,2],".")

# Annotate each point in the scatter plot
for i in 1:n
    annotate(dataTable[i+1,1],
	xy=[Z[i,1],Z[i,2]],
	xycoords="data")
end

# Calculate the variance explained by 2D representation
Error = (Z*model.W) - X
frobeniusNorm_Error = sum(Error.^2)
frobeniusNorm_X = sum(X.^2)

varianceRemaining = frobeniusNorm_Error / frobeniusNorm_X
@show varianceRemaining,k
