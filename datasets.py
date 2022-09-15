import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


class FoodKT(Dataset):
    def __init__(self, args, img_paths, labels=None, mode='train'):
        self.mode = mode
        self.img_size = args.img_size
        self.img_paths = img_paths
        self.labels = labels

        if self.mode == 'train':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'valid':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'test':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        img = cv2.imread(f'{img_path}')
        img = self.transform(image=img)

        if self.mode in ['train', 'valid']:
            label = self.labels[idx]
            return {'path': img_path, 'img': img['image'], 'label': torch.tensor(label)}

        else:
            img_name = img_path.split('/')[-1]
            return {'path': img_path, 'img': img['image'], 'img_name': img_name}
