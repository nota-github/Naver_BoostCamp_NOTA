import os
import torch
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2

def generate_loader(opt, phase=None):
    
    kwargs = {
        "batch_size": opt.batch_size if phase == "train" else 1,
        "num_workers": opt.num_workers if phase == "train" else 0,
        "shuffle": phase == "train",
    }
    dataset = Dataset(opt, phase)
    return torch.utils.data.DataLoader(dataset, **kwargs)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, phase):
        self.data_dir = opt.data_dir
        self.phase = phase
        self.img_dir = os.path.join(self.data_dir, 'image', phase)
        self.ann_dir = os.path.join(self.data_dir, 'mask', phase)
        self.file_name = sorted(os.listdir(self.img_dir))
        self.img_name = list(map(lambda x: os.path.join(self.img_dir, x), self.file_name))
        self.ann_name = list(map(lambda x: os.path.join(self.ann_dir, x.replace('jpg', 'png')), self.file_name))
        
        self.transform = self.train_transforms if phase == 'train' else self.val_transforms

        assert len(self.img_name) == len(self.ann_name), "There must be as many images as there are segmentation maps"

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_name[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        org_img = img.copy()
        h, w = img.shape[:2]
        label = cv2.imread(os.path.join(self.ann_dir, self.ann_name[idx]))
        if label.shape[2] == 3:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) # must be one channel
    
        out = self.transform(img, label)
        if self.phase == 'train':
            return {'pixel_values': out['image'], 'labels': out['mask'], 'name':self.file_name[idx]}
        else:
            return {'pixel_values': out['image'], 'labels': out['mask'], 'name':self.file_name[idx], 'org_img': org_img, 'org_shape': (h, w)}
        
    def __len__(self):
        return len(self.file_name)
    
    def train_transforms(self, img, label):
        crop_size = 512
        ratio_range = [0.5, 2.0]
        img_scale = [2048, 512]
        
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

        h, w = img.shape[:2]
        ratio = np.random.random_sample() * (ratio_range[1] - ratio_range[0]) + ratio_range[0]
        scale = (img_scale[0] * ratio, img_scale[1] * ratio)
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w)) 
        if (scale_factor * min(h, w)) < crop_size:
            scale_factor = crop_size / min(h, w)
        train_transforms = A.Compose([
                                A.Resize(int(h * scale_factor)+1, int(w * scale_factor)+1, p=1),
                                A.RandomCrop(crop_size, crop_size),
                                A.Flip(p=0.5),
                                A.ColorJitter(brightness=0.25, 
                                              contrast=0.25, 
                                              saturation=0.25, 
                                              hue=0.1, 
                                              p=0.5),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                A.PadIfNeeded(crop_size, crop_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
                                ToTensorV2()
                            ])
        out = train_transforms(image=img, mask=label)
        return out


    def val_transforms(self, img, label): 
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
        h, w = img.shape[:2]
        scale_factor = min(2048 / max(h, w), 512 / min(h, w)) 

        val_transforms = A.Compose([
                                A.Resize(int(h * scale_factor), int(w * scale_factor), p=1),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                            ])
        out = val_transforms(image=img, mask=label)
        return out
    