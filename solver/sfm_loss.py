#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn.functional as F
# from utils.keypoint_op import warp_points
# from utils.tensor_op import pixel_shuffle_inv



"""
param image1_keypoints: [[1,2],[3,4],[9,9]..] Batch*[keypoints]
param corresponding_keypoints: [[5,7],[3,2],[4,6]..]
"""
def descriptor_loss(img_kpts,img1_kpts,descriptors,descriptors1,keypointmap,keypointmap1):
    batch,_,H,W = descriptors.shape
    batch_corresponding =None
    for image,image1 in zip(img_kpts,img1_kpts):
        pair = [p1+p2 for p1,p2 in zip(image,image1)]
        corresponding = torch.sparse_coo_tensor(torch.Tensor(pair).t(),[1]*len(image),(H,W,H,W))
        corresponding = corresponding.unsqueeze(0)
        if batch_corresponding == None:
            batch_corresponding = corresponding
        else:
            batch_corresponding = torch.vstack((batch_corresponding,corresponding))

    # finally we got sparse corresponding matrix in (Batch,H,W,H,W) form. if the corresponding value is 1 means the correspondance

    descriptor  = descriptor*keypointmap #only leave keypoints for fast process [batch desc H w]
    descriptor1 = descriptors1* keypointmap1
    descriptor = descriptor.to_sparse()
    descriptor1 = descriptor1.to_sparse()

def descriptorScore_sparsematrix(descriptor,descriptor1,keypointmap,keypointmap1):
    batch_size,_,H,W = descriptor.shape
    print(descriptor.shape)
    descriptor = descriptor*keypointmap.unsqueeze(1)
    # descriptor = torch.reshape(descriptor, [batch_size, -1, H, W, 1, 1])
    descriptor = F.normalize(descriptor, p=2, dim=1)
    descriptor = descriptor.to_sparse()

    descriptor1 = descriptor1*keypointmap1.unsqueeze(1)
    # descriptor1 = torch.reshape(descriptor1, [batch_size, -1,  1, 1,H, W])
    descriptor1 = F.normalize(descriptor1, p=2, dim=1)
    descriptor1 = descriptor1.to_sparse()

    result = descriptor * descriptor1
    dot_product_desc = torch.sum(descriptor * descriptor1, dim=1)

# def descriptorScore_extractKeypoints(descriptor,img_kpts,descriptor1,img1_kpts):
#     batch,_,H,W = descriptor.shape
#     batch_descriptor = []
#     for x,y in img_kpts:
#         batch_descriptor.append(descriptor[x],[y])

if __name__=='__main__':
    des = torch.rand([2,256,240,320]).cuda()
    des1 = torch.rand([2,256,240,320]).cuda()
    numKeyPoints = 100
    batchSize = 2
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
    batch_keymap1 = None
    for i in range(batchSize):
        x = np.random.randint(240,size=numKeyPoints)
        y = np.random.randint(320,size=numKeyPoints)
        value = np.array([x,y])
        keymap1 = torch.sparse_coo_tensor(torch.Tensor(value),[i]*numKeyPoints,(240,320)).cuda()
        keymap1 = keymap1.unsqueeze(0)
        if batch_keymap1 == None:
            batch_keymap1 = keymap1
        else:
            batch_keymap1 = torch.vstack((batch_keymap1,keymap1))
    batch_keymap1 = batch_keymap1.to_dense()
    descriptorScore_sparsematrix(des,des1,batch_keymap,batch_keymap1)
    # print(i.shape)
