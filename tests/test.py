import torch
import torch.nn.functional as F
import time
import numpy as np
# a = torch.rand([240,320,240,320]).cuda()
b=torch.from_numpy(np.random.choice(2,10000)).reshape(100,100).cuda()
print(b.shape)
# b = b.to_sparse()
# c = torch.vstack((b,b))
# print(c.is_sparse)
# c = torch.sum(c, dim=1)
# print(c.shape)