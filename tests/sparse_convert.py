import torch
a = [[1,2],[3,4]]
b = [[2,6],[7,8]]

c = [p1+p1 for p1,p2 in zip(a,b)]
v = len(a)*[1]
import numpy as np 
c = np.array(c)
print(c.shape)
a = torch.sparse_coo_tensor(torch.Tensor(c).t(),v,(10,10,10,10))
a  =  a.unsqueeze(0)
a = 1