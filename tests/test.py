import torch
a = torch.rand([2,256,30,40]).cuda()
mask = torch.rand([2,30,40]).cuda()
mask = mask.unsqueeze(1)
result = a*mask
print(result)