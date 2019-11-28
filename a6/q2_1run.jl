# Load data
using DelimitedFiles
dataTable = readdlm("animals.csv",',')
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)

include("q2_1.jl")
k = 3
Z = MDS(X, k)

# Plot matrix as image
using PyPlot
figure(1)
clf()
plot(Z[:,1],Z[:,2],"b.")
for i in 1:n
    annotate(dataTable[i+1,1],
	xy=[Z[i,1],Z[i,2]],
	xycoords="data")
end
