import cv2
import numpy as np
import solver.descriptor_evaluation as ev
from utils.plt import plot_imgs

def draw_matches(data):
    keypoints1 = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in data['keypoints1']]
    keypoints2 = [cv2.KeyPoint(float(p[1]), float(p[0]), 1) for p in data['keypoints2']]
    inliers = data['inliers'].astype(bool)
    matches = np.array(data['matches'])[inliers].tolist()
#     good_matches = []
#     for match in matches:
#         if match.trainIdx < len(keypoints2) and match.queryIdx < len(keypoints1):
#             good_matches.append(match)
    img1 = cv2.merge([data['image1'], data['image1'], data['image1']]) * 255
    img2 = cv2.merge([data['image2'], data['image2'], data['image2']]) * 255
    return cv2.drawMatches(np.uint8(img1), keypoints1, np.uint8(img2), keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))
if __name__=='__main__':
    experiments = ['./data/descriptors/hpatches/sp/']
    num_images = 50
    for e in experiments:
        orb = True if e[:3] == 'orb' else False
        outputs = ev.get_homography_matches(e, keep_k_points=1000, correctness_thresh=3, num_images=num_images, orb=orb)
        for ind, output in enumerate(outputs):
            img = draw_matches(output) / 255.
            plot_imgs([img], titles=[e], dpi=200, save='/home/dev/SuperPoint-Pytorch/data/hpatches/'+str(ind)+'.png')
            
    print('Draw Done')