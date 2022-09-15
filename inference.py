import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FoodKT
from models import ImageModel


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader), ncols=120)

    preds, img_names = [], []
    for batch_idx, batch_item in batch_iter:
        img = batch_item['img'].to(args.device)
        img_name = batch_item['img_name']

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(img)
        preds.extend(torch.softmax(pred, dim=1).clone().detach().cpu().numpy())  # probabillity, not label
        img_names.extend(img_name)
    return preds, img_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--save_dir', type=str, default='/hdd/sy/weights/food-kt/submissions')
    parser.add_argument('-cv', '--csv_name', type=str, default='test')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        default='/hdd/sy/weights/food-kt/convnext_tiny_in22ft1k_0915_080814_fold_0/ckpt_best_fold_0.pt')
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-is', '--img_size', type=int, default=224)
    parser.add_argument('-nw', '--num_workers', type=int, default=8)
    parser.add_argument('-m', '--model', type=str, default='convnext_tiny_in22ft1k')
    parser.add_argument('--amp', type=bool, default=True)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #### SET DATASET ####
    label_description = sorted(os.listdir('/hdd/sy/food-kt/train'))
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    # test_data = sorted(glob('/hdd/sy/food-kt/train/*/*.jpg'))  # round1 test only
    test_data = sorted(glob('/hdd/sy/food-kt/test/*/*.jpg'))
    #####################

    #### LOAD DATASET ####
    test_dataset = FoodKT(args, test_data, labels=None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print('> DATAMODULE BUILT')
    ######################

    #### LOAD MODEL ####
    model = ImageModel(model_name=args.model, class_n=len(label_description), mode='test')
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    model = model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print('> MODEL BUILT')
    ####################

    #### INFERENCE START ####
    print('> START INFERENCE ')
    preds, img_names = test(model, test_loader)
    preds = np.argmax(preds, axis=1)
    preds = np.array([label_decoder[val] for val in preds])

    submission = pd.DataFrame()
    submission['image_name'] = img_names
    submission['label'] = preds

    submission.to_csv(f'{args.save_dir}/{args.csv_name}.csv', index=False)
    #########################
