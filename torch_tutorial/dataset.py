# Penn-Fudan Dataset for Pedestrian Detection and Segmentation (instance segmentation)
# Penn-Fudan Dataset contains 170 images with 345 instances of pedestrians
# This code below has strong dependece on this tutorial (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

# inst seg 같은 경우에는 클래스 중복이 되지 않을 것임. 따라서 같은 클래스의 위치를 찾고 

from torch.utils.data import Dataset
import os, glob
from PIL import Image
import numpy as np


class PennFudanDataset(Dataset):
    def __init__(self, root="data/PennFudanPed/"):
        super().__init__()
        self.root = root
        self.imgs = sorted(glob.glob(root+"/PNGImages/*.png"))
        self.masks = sorted(glob.glob(root+"/PedMasks/*.png")) # mask만 주어져있으면 instance segmentation은 따로 label을 읽을 필요가 없을 것 같은데?

    def __len__(self):
        return

    def __getitem__(self, idx):
        """
        Returns
            - img (numpy.array|PIL Image|torch.Tensor)
            - target (dictionary)
                - boxes (FloatTensor[N,4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H.
                - labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
                - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
                - area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
                - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
                - masks (UInt8Tensor[N, H, W]): Optional. The segmentation masks for each one of the objects
                - keypoints (FloatTensor[N, K, 3]): Optional. For each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt references/detection/transforms.py for your new keypoint representation
        """
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = np.array(Image.open(self.masks[idx]))
        

        
        return

