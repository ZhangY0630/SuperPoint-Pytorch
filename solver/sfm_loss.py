from calendar import c
from matplotlib import image
import numpy as np
import torch
import torch.nn.functional as F
from utils.keypoint_op import warp_points
from utils.tensor_op import pixel_shuffle_inv

def loss_func_sfm(config,data,prob,desc=None,prob_pair=None,desc_pair=None,device='cpu',):
        det_loss = detector_loss(data['image']['warp']['kpts_map'],
                             prob['logits'],
                             data['image']['warp']['mask'], 
                             config['grid_size'],
                             device=device)
        if desc is None or prob_pair is None or desc_pair is None:
            return det_loss 
        
        det_loss_warp = detector_loss(data['image1']['warp']['kpts_map'],
                            prob_pair['logits'],
                            data['image1']['warp']['mask'],
                            config['grid_size'],
                            device=device)


        validmask = torch.stack((data['image']['warp']['mask'],data['image1']['warp']['mask']),dim=1)

        # pairs = []
        # for batch in range(len(data['index'])):
        #     pairs.append( [pair for pair in zip(data['index'][batch],data['pair'][batch])])
        pairs = data['pairs']
        
        #descriptor_loss(config,img_kpts,img1_kpts,descriptors,descriptors1,pair,mask,device='cpu'):
        weighted_des_loss = descriptor_loss(config,
                            data['image']['warp']['kpts'],
                            data['image1']['warp']['kpts'],
                            desc['desc'],
                            desc_pair['desc'],
                            pairs,
                            validmask,
                            device)
        loss = det_loss + det_loss_warp + weighted_des_loss

        a, b, c = det_loss.item(), det_loss_warp.item(), weighted_des_loss.item()
        print('debug: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(a, b,c,a+b+c))
        return loss


def detector_loss(keypoint_map, logits, valid_mask=None, grid_size=8, device='cpu'):
    """
    :param keypoint_map: [B,H,W]
    :param logits: [B,65,Hc,Wc]
    :param valid_mask:[B, H, W]
    :param grid_size: 8 default
    :return:
    """
    # Convert the boolean labels to indices including the "no interest point" dustbin
    labels = keypoint_map.unsqueeze(1).float()#to [B, 1, H, W]
    labels = pixel_shuffle_inv(labels, grid_size) # to [B,64,H/8,W/8]
    B,C,h,w = labels.shape#h=H/grid_size,w=W/grid_size
    labels = torch.cat([2*labels, torch.ones([B,1,h,w],device=device)], dim=1)
    # Add a small random matrix to randomly break ties in argmax
    labels = torch.argmax(labels + torch.zeros(labels.shape,device=device).uniform_(0,0.1),dim=1)#B*65*Hc*Wc

    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(1)
    valid_mask = pixel_shuffle_inv(valid_mask, grid_size)#[B, 64, H/8, W/8]
    valid_mask = torch.prod(valid_mask, dim=1).unsqueeze(dim=1).type(torch.float32)#[B,1,H/8,W/8]

    ## method 1
    ce_loss = F.cross_entropy(logits, labels, reduction='none',)
    valid_mask = valid_mask.squeeze(dim=1)
    loss = torch.divide(torch.sum(ce_loss * valid_mask, dim=(1, 2)), torch.sum(valid_mask + 1e-6, dim=(1, 2)))
    loss = torch.mean(loss)

    ## method 2
    ## method 2 equals to tf.nn.sparse_softmax_cross_entropy()
    # epsilon = 1e-6
    # loss = F.log_softmax(logits,dim=1)
    # mask = valid_mask.type(torch.float32)
    # mask /= (torch.mean(mask)+epsilon)
    # loss = torch.mul(loss, mask)
    # loss = F.nll_loss(loss,labels)
    return loss

"""
:param image1_keypoints: [[1,2],[3,4],[9,9]..] Batch*[keypoints]
:param corresponding_keypoints: [[5,7],[3,2],[4,6]..]
"""
def descriptor_loss(config,img_kpts,img1_kpts,descriptors,descriptors1,pair,mask,device='cpu'):
    """
    :param image1_keypoints: [[1,2],[3,4],[9,9]..] Batch*[keypoints] B,N,2
    :param corresponding_keypoints: [[5,7],[3,2],[4,6]..] [B 256 H W]
    :param: pair -> [[image_keypoint_index,image1_keypoint_index],[7,4],...] [N,2] : corresponding size [[7,4]]
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
        pairlens = len(pair[idx])
        if pairlens==0:
            print("no pairs")
            continue
        valid_mask = torch.ones([pairlens,pairlens]).to(device)
        for ind, p in enumerate(pair[idx]):
            x = img[p[0]][0]
            y = img[p[0]][1]
            if (image_mask[x][y]==0):
                valid_mask[ind,:] = 0
            x = img1[p[1]][0]
            y = img1[p[1]][1]
            if(image1_mask[x][y]==0):
                valid_mask[:,ind]=0
        # for n,(x,y) in enumerate(img):
        #     if(image_mask[x][y]==0):
        #         valid_mask[n,:]=0
        # for n,(x,y) in enumerate(img1): 
        #     if(image1_mask[x][y]==0):
        #         valid_mask[:,n]=0

        normalization = torch.sum(valid_mask).to(device)


        dot_product_desc = descriptorScore_singleBatch(img,img1,choosen_descriptor,choosen_descriptor1,pair[idx])
        s = torch.eye(len(pair[idx]),device=device)
        # s = descriptorCorrespondence_singleBatch(imgKeyPointSize,img1KeyPointSize,pair[idx],device)
        positive_dist = torch.maximum(torch.tensor(0.,device=device), positive_margin - dot_product_desc)
        negative_dist = torch.maximum(torch.tensor(0.,device=device), dot_product_desc - negative_margin)
        loss = (lambda_d * s * positive_dist + (1 - s) * negative_dist).to(device)

        loss = lambda_loss*torch.sum(valid_mask * loss)/normalization

        total_loss = total_loss+loss
    return total_loss


def precision_recall(pred, keypoint_map, valid_mask):
    pred = valid_mask * pred
    labels = keypoint_map

    precision = torch.sum(pred*labels)/torch.sum(pred)
    recall = torch.sum(pred*labels)/torch.sum(labels)

    return {'precision': precision, 'recall': recall}


# torch.bmm
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
def descriptorScore_singleBatch(img,img1,descriptor,descriptor1,pair):
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
    descriptors1 = None
    for p in pair:
        img_idx = p[0]
        x = img[img_idx][0]
        y = img[img_idx][1]
    
        kpt = descriptor[x][y]
        if descriptors ==None:
            descriptors = kpt
        else:
            descriptors = torch.vstack((descriptors,kpt))
            
        
        img_idx = p[1] 
        x = img1[img_idx][0]
        y = img1[img_idx][1]
        kpt1 = descriptor1[x][y]
        if descriptors1 ==None:
            descriptors1 = kpt1
        else:
            descriptors1 = torch.vstack((descriptors1,kpt1))
            
            
    descriptors = descriptors.reshape([1,-1,len(pair),1])
    descriptors = F.normalize(descriptors, p=2, dim=1)
    


    descriptors1 = descriptors1.reshape([1,-1,1,len(pair)])
    descriptors1 = F.normalize(descriptors1, p=2, dim=1)

    dot_product_desc = torch.sum(descriptors*descriptors1,dim=1).squeeze(0)
    dot_product_desc = F.relu(dot_product_desc)

    return dot_product_desc
def descriptorCorrespondence_singleBatch(kptSize,kptSize1,pair,device='cpu'):
    """
    :param: kptSize -> int :image keypoints size
    :param: kptSize1 -> int : image1 keypoints size
    :param: pair -> [[2,5],[7,4],...] [N,2] : corresponding size
    """
    corres = torch.sparse_coo_tensor(pair.t(),torch.as_tensor([1]*len(pair), device = device),(len(pair),len(pair)), device = device)
    # corres = torch.sparse_coo_tensor(torch.tensor(pair).t(),[1]*len(pair),(kptSize,kptSize1))
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