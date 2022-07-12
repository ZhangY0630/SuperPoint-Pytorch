#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn.functional as F
# from utils.keypoint_op import warp_points
# from utils.tensor_op import pixel_shuffle_inv



"""
:param image1_keypoints: [[1,2],[3,4],[9,9]..] Batch*[keypoints]
:param corresponding_keypoints: [[5,7],[3,2],[4,6]..]
"""
def descriptor_loss(config,img_kpts,img1_kpts,descriptors,descriptors1,pair,mask,device='cpu'):
    """
    :param image1_keypoints: [[1,2],[3,4],[9,9]..] Batch*[keypoints]
    :param corresponding_keypoints: [[5,7],[3,2],[4,6]..]
    :param: pair -> [[image_keypoint_index,image1_keypoint_index],[7,4],...] [N,2] : corresponding size
    :param: mask ->[batch,2,H,W]
    """

    positive_margin = config['loss']['positive_margin']
    negative_margin = config['loss']['negative_margin']
    lambda_d = config['loss']['lambda_d']
    lambda_loss = config['loss']['lambda_loss']

    batch,_,H,W = descriptors.shape
    total_loss = 0
    for idx in range(batch):
        img = img_kpts[idx]
        imgKeyPointSize = len(img)
        img1 = img1_kpts[idx]
        img1KeyPointSize = len(img1)
        choosen_descriptor = descriptors[idx]
        choosen_descriptor1 = descriptors1[idx]

        image_mask = mask[idx][0]
        image1_mask = mask[idx][1]
        valid_mask = torch.ones([imgKeyPointSize,img1KeyPointSize])
        for n,(x,y) in enumerate(img):
            if(image_mask[x][y]==0):
                valid_mask[n,:]=0
        for n,(x,y) in enumerate(img1): 
            if(image1_mask[x][y]==0):
                valid_mask[:,n]=0

        normalization = torch.sum(valid_mask)


        dot_product_desc = descriptorScore_singleBatch(img,img1,choosen_descriptor,choosen_descriptor1)
        s = descriptorCorrespondence_singleBatch(imgKeyPointSize,img1KeyPointSize,pair)
        positive_dist = torch.maximum(torch.tensor(0.,device=device), positive_margin - dot_product_desc)
        negative_dist = torch.maximum(torch.tensor(0.,device=device), dot_product_desc - negative_margin)
        loss = lambda_d * s * positive_dist + (1 - s) * negative_dist

        loss = lambda_loss*torch.sum(valid_mask * loss)/normalization
        total_loss = total_loss+loss
    return total_loss


# Reason to discard this:
# 1. can't find a way to generate sparse matrix as multiplication result
# 1. don't have enough space to calculation
# def descriptorScore_sparsematrix(descriptor,descriptor1,keypointmap,keypointmap1):
#     batch_size,_,H,W = descriptor.shape
#     print(descriptor.shape)
#     descriptor = descriptor*keypointmap.unsqueeze(1)
#     # descriptor = torch.reshape(descriptor, [batch_size, -1, H, W, 1, 1])
#     descriptor = F.normalize(descriptor, p=2, dim=1)
#     descriptor = descriptor.to_sparse()

#     descriptor1 = descriptor1*keypointmap1.unsqueeze(1)
#     # descriptor1 = torch.reshape(descriptor1, [batch_size, -1,  1, 1,H, W])
#     descriptor1 = F.normalize(descriptor1, p=2, dim=1)
#     descriptor1 = descriptor1.to_sparse()

#     result = descriptor * descriptor1 #not support
#     dot_product_desc = torch.sum(descriptor * descriptor1, dim=1)
#----------------------------------------------------------------------------
# Reason to discard this:
# 1. keypoint size varies,can't form uniform matrix
# def descriptorScore_extractKeypoints(descriptor,img_kpts,descriptor1,img_kpts1):
#     batch_size,_,H,W = descriptor.shape
#     descriptor = descriptor.reshape([batch_size,H,W,-1])
#     descriptor1 = descriptor1.reshape([batch_size,H,W,-1])
#     batch_descriptor = None
#     #initialise for image
#     for idx,image in enumerate(img_kpts):
#         descriptors = None
#         for x,y in image:
#             kpt = descriptor[idx][x][y]
#             if descriptors ==None:
#                 descriptors = kpt
#             else:
#                 descriptors = torch.vstack((descriptors,kpt))
#         descriptors=descriptors.unsqueeze(0)
#         if batch_descriptor ==None:
#             batch_descriptor = descriptors
#         else:
#             batch_descriptor = torch.vstack((batch_descriptor,descriptors))
#     #initialise for image1
#     batch_descriptor1 = None
#     for idx,image in enumerate(img_kpts1):
#         descriptors1 = None
#         for x,y in image:
#             kpt1 = descriptor1[idx][x][y]
#             if descriptors1 ==None:
#                 descriptors1 = kpt1
#             else:
#                 descriptors1 = torch.vstack((descriptors1,kpt1))
#         descriptors1=descriptors1.unsqueeze(0)
#         if batch_descriptor1 ==None:
#             batch_descriptor1 = descriptors1
#         else:
#             batch_descriptor1 = torch.vstack((batch_descriptor1,descriptors1))

#     batch_descriptor = torch.reshape(batch_descriptor, [batch_size, -1, len(img_kpts), 1])
#     print(batch_descriptor.shape)
#---------------------------
def descriptorScore_singleBatch(img,img1,descriptor,descriptor1):
    """
    :param: img -> 2xN list 
    :param: img1 -> 2xM list
    :param: descriptor -> [H,W,descriptor_size]
    :param: descriptor1 -> [H,W,descriptor_size]

    return [NxM] matching score
    """
    descriptor_dim,H,W = descriptor.shape
    descriptor = descriptor.reshape([H,W,descriptor_dim])
    descriptor1 = descriptor1.reshape([H,W,descriptor_dim])

    descriptors = None
    for x,y in img:
        kpt = descriptor[x][y]
        if descriptors ==None:
            descriptors = kpt
        else:
            descriptors = torch.vstack((descriptors,kpt))
    descriptors = descriptors.reshape([1,-1,len(img),1])
    descriptors = F.normalize(descriptors, p=2, dim=1)
    print(descriptors.shape)
    descriptors1 = None
    for x,y in img1:
        kpt1 = descriptor1[x][y]
        if descriptors1 ==None:
            descriptors1 = kpt1
        else:
            descriptors1 = torch.vstack((descriptors1,kpt1))
    descriptors1 = descriptors1.reshape([1,-1,1,len(img1)])
    descriptors1 = F.normalize(descriptors1, p=2, dim=1)
    result = descriptors*descriptors1
    dot_product_desc = torch.sum(result,dim=1).squeeze(0)
    dot_product_desc = F.relu(dot_product_desc)
    return dot_product_desc
def descriptorCorrespondence_singleBatch(kptSize,kptSize1,pair):
    """
    :param: kptSize -> int :image keypoints size
    :param: kptSize1 -> int : image1 keypoints size
    :param: pair -> [[2,5],[7,4],...] [N,2] : corresponding size
    """
    corres = torch.sparse_coo_tensor(torch.tensor(pair).t(),[1]*len(pair),(kptSize,kptSize1))
    corres = corres.to_dense()
    return corres


if __name__=='__main__':
    des = torch.rand([256,240,320]).cuda()
    des1 = torch.rand([256,240,320]).cuda()
    numKeyPoints = 100
    batchSize = 2


    x = np.random.randint(240,size=numKeyPoints)
    y = np.random.randint(320,size=numKeyPoints)
    kpts= [out for out in zip(x,y)]
    

    x = np.random.randint(240,size=numKeyPoints)
    y = np.random.randint(320,size=numKeyPoints)
    kpts1= [out for out in zip(x,y)]

    descriptorScore_singleBatch(kpts,kpts1,des,des1)

    x = np.random.randint(100,size=20)
    y = np.random.randint(100,size=30)
    pair =[out for out in zip(x,y)]
    descriptorCorrespondence_singleBatch(100,100,pair)


