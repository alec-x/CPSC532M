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
k = 2
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

# Find the trait of animals that has largest influence on 1st principal component
index_PC1 = argmax(abs.(model.W[1,:]))
trait_PC1 = dataTable[1,index_PC1 + 1]
@show trait_PC1

# Find the trait of animals that has largest influence on 2nd principal component
index_PC2 = argmax(abs.(model.W[2,:]))
trait_PC2 = dataTable[1,index_PC2 + 1]
@show trait_PC2
