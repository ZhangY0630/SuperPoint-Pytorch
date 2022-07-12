import torch
import numpy as np
import torch.nn.functional as F
a = torch.rand([8,256,240,320]).cuda()
batchSize = 8
numKeyPoints = 100
batch_keymap = None
for i in range(batchSize):

    x = np.random.randint(240,size=numKeyPoints)
    y = np.random.randint(320,size=numKeyPoints)
    value = np.array([x,y])
    keymap = torch.sparse_coo_tensor(torch.Tensor(value),[i]*numKeyPoints,(240,320)).cuda()
    keymap = keymap.unsqueeze(0)
    if batch_keymap == None:
        batch_keymap = keymap
    else:
        batch_keymap = torch.vstack((batch_keymap,keymap))
batch_keymap = batch_keymap.to_dense()
c = a*batch_keymap.unsqueeze(1)

c  = c.reshape([8,76800,256])
cs = c.reshape([8,256,76800])
c = c.to_sparse()

out = torch.bmm(c,cs)


# b=torch.from_numpy(np.random.choice(2,157286400)).reshape(8,76800,256).cuda()
# c = a*b
# c = F.normalize(c,p=2,dim=2)
# cs = c.to_sparse()
# print(cs.shape)



# a1 = torch.rand([8,256,76800]).cuda()
# b1=torch.from_numpy(np.random.choice(2,157286400)).reshape(8,256,76800).cuda()
# c1 = a1*b1
# c1 = F.normalize(c1,p=2,dim=1)
# print(c1.shape)
# out = torch.bmm(cs,c1)
# print(out.is_sparse)
# # # c*a1
# # torch.bmm(c*c1)
# # torch.bmm(cs*c1)
# # # torch.bmm(c*c1)

# # a = torch.randn(2, 1200, 256).to_sparse().requires_grad_(True)
# # b = torch.randn(2, 256, 1200)
# # c = torch.bmm(a,b)
# # c = c.reshape([2,30,40,30,40])
# # print(c.shape)