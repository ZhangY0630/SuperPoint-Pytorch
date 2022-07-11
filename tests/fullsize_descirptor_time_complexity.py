import torch
import torch.nn.functional as F
import time
import numpy as np
# a = torch.rand([8,256,int(240/8),int(320/8),1,1]).cuda()
# b = torch.rand([8,256,1,1,int(240/8),int(320/8)]).cuda()
# c = a*b
# a = torch.rand([8,72,1256])
# b = torch.rand([8,256,82])

# c = torch.bmm(a,b).unsqueeze(1)
# c = torch.bmm(a,b)
# print(c[0][0][0])

# a = torch.rand([8,256,1,72])
# a = F.normalize(a,p=2,dim=1)
# b = torch.rand([8,256,82,1])
# b = F.normalize(b,p=2,dim=1)
# c =a*b
# d = torch.sum(c,dim=1)
# print(d[0][0][0])

# a=torch.from_numpy(np.random.choice(2,80)).reshape(4,4,5)
# des = torch.rand((4,4,5))
# c = des*a
# d = c.flatten()
# e = c.to_sparse()
# print(e)
# e = torch.nonzero(d)
# print(d)
# print(e)

a = torch.rand([8,30,40,256]).cuda()
b=torch.from_numpy(np.random.choice(2,2457600)).reshape(8,30,40,256).cuda()
c = a*b
c = F.normalize(c,p=2,dim=1)
d = c.to_sparse()


a1 = torch.rand([8,256,30,40]).cuda()
b1=torch.from_numpy(np.random.choice(2,2457600)).reshape(8,256,30,40).cuda()
c1 = a1*b1
c1 = F.normalize(c1,p=2,dim=1)
d1 = c

torch.sparse.mm(c1,d)

result = torch.matmul(d,d1)
# print(result.is_sparse)

# a = torch.rand([1,256,3,4]).cuda()
# b=torch.from_numpy(np.random.choice(2,256*3*4)).reshape(1,256,3,4).cuda()
# c = a*b
# c.unsqueeze(1)
# c = F.normalize(c,p=2,dim=1)
# d = c.to_sparse()

# [1,2] [3,4]c | [7,4],[2,3] <- [8*8]

# i = [[1 ,7],
#     [2 ,4],
#     [3,2 ],
#     [4,3]]
# v =[1,1]

# s = torch.sparse_coo_tensor(i,v,(8,8,8,8))
# i = [[1,2,3,4],[7,4,2,3]]
# a = torch.sparse_coo_tensor(torch.tensor(i).t(),v,(8,8,8,8))
# print(a)
# print(a[1][2][3][5])

