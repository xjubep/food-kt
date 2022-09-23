import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


class FoodKT(Dataset):
    def __init__(self, args, img_paths, labels=None, mode='train'):
        self.aug_ver = args.aug_ver
        self.mode = mode
        self.img_size = args.img_size
        self.img_paths = img_paths
        self.labels = labels

        if self.mode == 'train':
            if self.aug_ver == 0:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                        scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                        interpolation=1, p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                       interpolation=1, border_mode=0, value=0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.OneOf([
                        A.CLAHE(clip_limit=2, p=1.0),
                        A.Sharpen(p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    ], p=0.25),
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])
            elif self.aug_ver == 1:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                        scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                        interpolation=1, p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                       interpolation=1, border_mode=0, value=0, p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.OneOf([
                        A.CLAHE(clip_limit=2, p=1.0),
                        A.Sharpen(p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    ], p=0.25),
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])
            elif self.aug_ver == 2:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                        scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                        interpolation=1, p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                       interpolation=1, border_mode=0, value=0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf([
                        A.CLAHE(clip_limit=2, p=1.0),
                        A.Sharpen(p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    ], p=0.25),
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])
            elif self.aug_ver == 3:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                        scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                        interpolation=1, p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                       interpolation=1, border_mode=0, value=0, p=0.5),
                    A.OneOf([
                        A.CLAHE(clip_limit=2, p=1.0),
                        A.Sharpen(p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    ], p=0.25),
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])
            elif self.aug_ver == 4:
                self.transform = A.Compose([
                    A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                        scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                        interpolation=1, p=1.0),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                       interpolation=1, border_mode=0, value=0, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])
            elif self.aug_ver == 5:
                self.transform = A.Compose([
                    # A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                    #                     scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                    #                     interpolation=1, p=1.0),
                    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                    #                    interpolation=1, border_mode=0, value=0, p=0.5),
                    # A.HorizontalFlip(p=0.5),
                    # A.VerticalFlip(p=0.5),
                    # A.OneOf([
                    #     A.CLAHE(clip_limit=2, p=1.0),
                    #     A.Sharpen(p=1.0),
                    #     A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    # ], p=0.25),
                    # A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    # ToTensorV2()
                ])
            elif self.aug_ver == 6:
                self.transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.OneOf([
                        A.CLAHE(clip_limit=2, p=1.0),
                        A.Sharpen(p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                    ], p=0.25),
                    A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2()
                ])
        elif self.mode == 'valid':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
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


