import numpy as np
a= [[1,2,3,4],[5,6,7,8]] 
b = [[2,3,4,5],[6,32,41,2]] 
output = []
for batch in range(len(a)):
    output.append( [pair for pair in zip(a[batch],b[batch])])

output = np.array(output)
print(output.shape)
