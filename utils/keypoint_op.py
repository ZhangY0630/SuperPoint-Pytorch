#-*-coding:utf8-*-
import collections
import numpy as np
import torch


def filter_points(points, shape, device='cpu', id_included = False):
    """
    :param points: (N,2), formated as (y, x)
    :param shape: (H, W)
    :return: filtered point without outliers
    """

    if len(points)!=0 and id_included == True:
        mask = (points[:, :2] >= torch.tensor([0,0], device=device)) & (points[:, :2] <= torch.tensor(shape, device=device)-1)
        mask = torch.all(mask, dim=1)
        return points[mask]
    elif len(points)!=0 and id_included == False:
        mask = (points >= 0) & (points <= torch.tensor(shape, device=device)-1)
        mask = torch.all(mask, dim=1)
        return points[mask]
    else:
        return points


def compute_keypoint_map(points, shape, device='cpu', id_included = False):
    """
    :param shape: (H, W)
    :param points: (N,2)
    :return:
    """
    if id_included:
        points = torch.as_tensor((points.numpy()[:,:-1]).astype(np.float32), device=device)
    coord = torch.minimum(torch.round(points).type(torch.int), torch.tensor(shape,device=device)-1)
    kmap = torch.zeros((shape),dtype=torch.int, device=device)
    kmap[coord[:,0],coord[:,1]] = 1
    return kmap


def warp_points(points, homographies, device='cpu', id_included = False):
    """
    :param points: (N,2), tensor
    :param homographies: [B, 3, 3], batch of homographies
    :return: warped points B,N,2
    """
    keys = None
    if id_included:
        keys = torch.as_tensor((points.numpy()[:,2]).astype(np.float32), device=device)
        points = torch.as_tensor((points.numpy()[:,:-1]).astype(np.float32), device=device)
        # print('warp_points points', points)
        # print('warp_points keys', keys)
    if len(points)==0:
        return points
    #TODO: Part1, the following code maybe not appropriate for your code
    points = torch.fliplr(points)
    if len(homographies.shape)==2:
        homographies = homographies.unsqueeze(0)
    B = homographies.shape[0]
    ##TODO: uncomment the following line to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    points = torch.cat((points, torch.ones((points.shape[0], 1),device=device)),dim=1)
    ##each row dot each column of points.transpose
    warped_points = torch.tensordot(homographies, points.transpose(1,0),dims=([2], [0]))#batch dot
    ##
    warped_points = warped_points.reshape([B, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    #TODO: Part2, the flip operation is combinated with Part1
    warped_points = torch.flip(warped_points,dims=(2,))
    #TODO: Note: one point case
    warped_points = warped_points.squeeze(dim=0)
    if id_included:
        warped_points = torch.cat((warped_points, keys.reshape([len(keys),1])), 1)
        # print('warp_points points', points)
    return warped_points

if __name__=='__main__':
    points = torch.tensor([[1.2,0.8],[13,9],[5.6,0.3],[0.4,0.8]])
    homographies = torch.tensor([[[1,0.5,0],[-0.5,1,0],[0,0,1]],[[0.3,0.84,0],[-0.5,1,0],[0,0,1]]])
    pt = warp_points(points, homographies, device='cpu')
    print(pt)
