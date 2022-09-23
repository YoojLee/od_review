# Penn-Fudan Dataset for Pedestrian Detection and Segmentation (instance segmentation)
# Penn-Fudan Dataset contains 170 images with 345 instances of pedestrians
# This code below has strong dependece on this tutorial (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

# For instance segmentation, there are no duplicates in classes. Therefore, for bounding boxes, if we find max and min values of a certain class, that would be the coords of bounding boxes.

import torch
from torch.utils.data import Dataset
import os, glob
from PIL import Image
import numpy as np
import torchvision.transforms as T


class PennFudanDataset(Dataset):
    def __init__(self, root="data/PennFudanPed/", transforms=None):
        super().__init__()
        self.root = root
        self.imgs = sorted(glob.glob(root+"/PNGImages/*.png"))
        self.masks = sorted(glob.glob(root+"/PedMasks/*.png")) # mask만 주어져있으면 instance segmentation은 따로 label을 읽을 필요가 없을 것 같은데?
        
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

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
        mask = np.array(Image.open(self.masks[idx])) # As opening the image with opencv, ensure that a image shape shoud be in an order of (c,h,w), not (h,w,c).
        
        # mask에서 instance를 구분하는 형식은 아니고, box를 따로 만들어서 거기서 instance 구분하는 방식임.
        obj_ids = np.unique(mask) # Get the object ids (each ids are unique.)
        obj_ids = obj_ids[1:] # array with N elements

        # split the color-encoded mask into a set of binary masks
        masks = mask = obj_ids[:, None, None] # By this, add new axis at an axis position 2, 3 (masks: [N,H,W])

        num_objs = len(obj_ids)
        boxes = []
        
        # Get the coords of bounding boxes for each instances.
        for i in range(num_objs):
            inst_pts = np.where(masks[i])
            xmin = np.min(inst_pts[1])
            xmax = np.max(inst_pts[1])
            ymin = np.min(inst_pts[0])
            ymax = np.max(inst_pts[0])
            boxes.append([xmin,xmax,ymin,ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1] * boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64) # suppose all instances are not crowd

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
            }

        if self.transforms is not None:
            img, target = self.transfroms(img, target)
        
        return img, target


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = PennFudanDataset()
    print(f"The dataset contains {len(dataset)} elements.")

    def collate_fn(batch):
        return tuple(zip(*batch))

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    print(next(iter(dataloader)))