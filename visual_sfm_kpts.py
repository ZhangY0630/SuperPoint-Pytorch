import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
filelist = glob.glob("../lvdi_/images/*")
filelist = ["../lvdi_/images/615947835.png"]
for img in filelist:
    name  = img.split(".png")[0]
    name = name.split('/')[-1]

    npy = "../lvdi_/keypoints/"+name+".npy"
    kpts = np.load(npy)
    im = cv2.imread(img)
    print(im.shape)
    for kpt in kpts:
        cv2.circle(im, (int(kpt[0]), int(kpt[1])), radius=3, color=(0, 255, 0))

    plt.imshow(im)
    plt.show()
    

    

    break