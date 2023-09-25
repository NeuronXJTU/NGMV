import glob
import numpy as np
import torch
import os
import cv2
from utils.common import *
import SimpleITK as sitk
import config
from unet_model import UNet
# import segmentation_models_pytorch as smp

from model.ConRes2 import ConResNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
from load_dataset_3D import *
from torch.utils.data import DataLoader
import cv2
import torch.nn as nn

from ce_net import cenet
from ce_net import att_unet
from collections import OrderedDict

from model.EffUNet import EffUNet
from model.DeepLabCon import DeepLabCon
from model.deeplabv3_plus2 import DeepLabv3_plus


def load_model(model, model_path):
    print(model_path)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['net']
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    args = config.args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConResNet((256, 256, args.batch_size), num_classes=args.num_classes, weight_std=True)
    net = load_model(net, '{}/best_model_1.pth'.format(args.save))
    net = net.cuda()
    net.eval()
    test_set = BasicDataset_brats(args.dataset_path_1, args.test_txt, 'c')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)


    for d in tqdm(test_loader, total=len(test_loader)):
        img = d["img"]
        res = d["res"]
        save_res_path = (d["mask_name"][0].split('/')[-1])
        res = res.float()
        res = res.cuda()
        img = img.float()
        img = img.cuda()

       # print(img.shape,res.shape)
        pred = net([img,res])
        pred = torch.sigmoid(pred[0])

        pred = np.array(pred.data.cpu()[0])
        img = np.array(img.data.cpu()[0])


        pred = np.array(pred)
        img=np.array(img)
        tifffile.imwrite('temp_m/pred.tiff', pred)
        tifffile.imwrite('temp_m/img.tiff', img)

        # print(pred.shape)

        yuzhi = 0.1
        pred[pred >= yuzhi] = 1
        pred[pred < yuzhi] = 0
        # print(pred.shape)

        pred_new = np.zeros((pred.shape[1], pred.shape[2]))
        # print(pred_new.shape)
        pred_new[pred[0] == 1] = 1
        #  pred_new[pred[1] == 1] = 2
        #        pred_new[pred[2] == 1] = 3
        #        pred_new[pred[3] == 1] = 4
        count = args.save.split('/')[-1].replace('model', '')
        file_name = d["mask_name"][0].split('/')[-2].replace('data', count)
        result_save_path = args.save_nii + file_name + '_' + args.name
        if not os.path.exists(result_save_path): os.mkdir(result_save_path)
        mask = sitk.GetImageFromArray(pred_new.astype(np.uint8))
        sitk.WriteImage(mask, os.path.join(result_save_path, save_res_path))

