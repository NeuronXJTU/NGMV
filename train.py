from torch.utils.data import DataLoader
import torch, os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
import torch.optim as optim
import config

from utils import init_util, metrics, common
from tqdm import tqdm
from collections import OrderedDict
from load_dataset_3D import *

import torch.nn as nn
from torch.autograd import Variable
import random
import tifffile
import numpy as np
import gc

np.seterr(divide='ignore', invalid='ignore')
import losses
import ramps
import time
from model.ConRes2 import ConResNet

import torch.multiprocessing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.multiprocessing.set_sharing_strategy('file_system')


def train_data(model, optimizer, iter_num, data, target,res):
    data, target,res = Variable(data), Variable(target), Variable(res)
    data,res = data.float(),res.float()
    data, target,res = data.cuda(0), target.cuda(),res.cuda()
    preds = model([data,res])
    output_seg=preds[0]
    preds_res = preds[1]

    preds_res = torch.sigmoid(preds_res)
    res = torch.sigmoid(res)
    outputs_soft = torch.sigmoid(output_seg-res)


    pred=outputs_soft[:]
    res_pred=preds_res[:]
    bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50])).cuda()
    supervised_loss = bce1(output_seg, target)
    loss_res = nn.MSELoss().cuda()
    loss_cr=loss_res(res,preds_res)

    loss = supervised_loss+loss_cr/args.batch_size
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    iter_num = iter_num + 1
    return (loss, supervised_loss, iter_num, pred,res_pred)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def load_model(model, model_path):
    print(model_path)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['net']
    start_epoch = checkpoint['epoch']

    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model,start_epoch


def train(iter_num_1, iter_num_2):
    global pred_1
    common.adjust_learning_rate(optimizer_1, epoch, args)
    common.adjust_learning_rate(optimizer_2, epoch, args)
    print("=======Epoch:{}=======".format(epoch))
    model_1.train()
    model_2.train()
    loss_1_2D, loss_2_2D, loss_labeled_3d, loss_unlabeled_3d = 0, 0, 0, 0
    temp_p_l = []
    print('------------c begin----------------')
    for idx, d in tqdm(enumerate(train_loader_1), total=len(train_loader_1)):

        data_1 = d["img"]
        target_1 = d["cate"]
        name_1 = d["mask_name"]
        res_1=d["res"]
        loss_1, supervised_loss_1_2D, iter_num_1, preds_1,preds_res_1 = train_data(model_1, optimizer_1,
                                                                      iter_num_1, data_1,
                                                                      target_1, res_1,
                                                                                  )

        loss_1_2D += loss_1


        for i in range(args.labeled_bs):
            p = [{'name': name_1[i], 'pred': preds_1[i], 'res': preds_res_1[i]}]
            temp_p_l.extend(p)

        del p
        gc.collect()
        torch.cuda.empty_cache()


        del data_1, target_1, preds_1,preds_res_1,res_1
        gc.collect()
        torch.cuda.empty_cache()

    loss_1_2D /= len(train_loader_1)
    p = sorted(temp_p_l, key=lambda i: i['name'])

    loss_1 = loss_1_2D


    l_pred_1 = []

    for i in range(len(p)):
        l_pred_1.extend([p[i]['pred']])

    l_pred_1 = torch.cat([torch.unsqueeze(i, 0) for i in l_pred_1], 0)
    del loss_1_2D,p
    torch.cuda.empty_cache()
    gc.collect()
    print('------------a begin----------------')
    temp_p_l = []
    for idx, d in tqdm(enumerate(train_loader_2), total=len(train_loader_2)):

        data_2 = d["img"]
        target_2 = d["cate"]
        name_2 = d["mask_name"]
        res_2 = d["res"]

        loss_2, supervised_loss_2_2D, iter_num_2, preds_2,preds_res_2 = train_data(model_2,
                                                                      optimizer_2,
                                                                      iter_num_2, data_2,
                                                                      target_2, res_2)

        loss_2_2D += loss_2



        for i in range(args.labeled_bs):
            p = [{'name': name_2[i], 'pred': preds_2[i],'res':preds_res_2[i]}]
            temp_p_l.extend(p)

        del data_2, target_2, preds_2,preds_res_2,p,res_2
        gc.collect()
        torch.cuda.empty_cache()

    loss_2_2D /= len(train_loader_2)
    p = sorted(temp_p_l, key=lambda i: i['name'])
    loss_2 = loss_2_2D
    l_pred_2 = []
    l_res_2 = []
    for i in range(len(p)):
        l_pred_2.append(p[i]['pred'])

    print('------------ready to 2D to 3D---------------')

    l_pred_2 = torch.cat([torch.unsqueeze(i, 0) for i in l_pred_2], 0)

    print('before: ', l_pred_1.shape, l_pred_2.shape)

    print('---------a to c----------------')

    #    a, c = 256, 512
  #  a, c = 320, 512
    a,c=256,128
   # a, c = 224, 448



    l_pred_11 = torch.transpose(l_pred_1[:c], 0, 1)
    l_pred_21 = l_pred_2[:a]
    l_pred_21 = l_pred_21.permute(1, 3, 0, 2)

    print('after: ', l_pred_11.shape, l_pred_21.shape)

    count_l = len(l_pred_1) // c
    print(count_l)

    for i in range(0, count_l):
        start_1 = i * c
        end_1 = (i + 1) * c
        temp_pred_1 = l_pred_1[start_1: end_1]
        start_2 = i * a
        end_2 = (i + 1) * a
        temp_pred_2 = l_pred_2[start_2: end_2]
        print('split: ', temp_pred_1.shape, temp_pred_2.shape)

        temp_pred_1 = torch.transpose(temp_pred_1, 0, 1)
        temp_pred_2 = temp_pred_2.permute(1, 3, 0, 2)
        print('after: ', temp_pred_1.shape, temp_pred_2.shape)
        l = torch.mean((temp_pred_1 - temp_pred_2) ** 2)


        loss_labeled_3d += l

    loss_labeled_3d=1*loss_labeled_3d

    train_loss = loss_1 + loss_2 + loss_labeled_3d

    train_loss_3d = loss_labeled_3d
    train_loss_3d = Variable(train_loss_3d, requires_grad=True)

    del l_pred_11, l_pred_21, temp_pred_1, temp_pred_2
    torch.cuda.empty_cache()

    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    train_loss_3d.backward()
    optimizer_1.step()
    optimizer_2.step()

    print('-----------loss------------------')
    print('train_loss, loss_1, loss_2, loss_labeled_3d, train_loss_3d', train_loss.item(),
          loss_1.item(), loss_2.item(), loss_labeled_3d.item(),
          train_loss_3d.item())

    state_1 = {'net': model_1.state_dict(), 'optimizer': optimizer_1.state_dict(), 'epoch': epoch}
    state_2 = {'net': model_2.state_dict(), 'optimizer': optimizer_2.state_dict(), 'epoch': epoch}
    if (best[1] > train_loss.item()):
        print('Saving best model')
        best[0] = epoch
        best[1] = train_loss.item()
        torch.save(state_1, os.path.join(args.save, f'best_model_1.pth'))
        torch.save(state_2, os.path.join(args.save, f'best_model_2.pth'))
    print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
    if epoch == args.epochs:
        torch.save(state_1, os.path.join(args.save, f'epoch{epoch}_1.pth'))
        torch.save(state_2, os.path.join(args.save, f'epoch{epoch}_2.pth'))
    return iter_num_1, iter_num_2


if __name__ == '__main__':
    iter_num_1 = 0
    iter_num_2 = 0
    start_epoch = 1
    train_set_1 = BasicDataset_brats(args.dataset_path_1, args.label_txt_1, 'c')
    print('dataset_1:', len(train_set_1))
    train_loader_1 = DataLoader(train_set_1, batch_size=args.batch_size, num_workers=16, pin_memory=False,
                                worker_init_fn=worker_init_fn)

    # 视角二
    train_set_2 = BasicDataset_brats(args.dataset_path_2, args.label_txt_2, 'a')
    print('dataset_2:', len(train_set_2))
    train_loader_2 = DataLoader(train_set_2, batch_size=args.batch_size, num_workers=16, pin_memory=False,
                                worker_init_fn=worker_init_fn)
    # 视角一
    check_path_1 = os.path.join(args.save, f'best_model_1.pth')
    check_path_2 = os.path.join(args.save, f'best_model_2.pth')
    if os.path.exists(check_path_1):
        print("从上次中断开始训练")
        model_1 = ConResNet((256, 256, args.batch_size), num_classes=args.num_classes, weight_std=True)
        model_1, start_epoch_1 = load_model(model_1, '{}/best_model_1.pth'.format(args.save))
        model_1 = nn.DataParallel(model_1)
        model_1 = model_1.cuda()
        model_2 = ConResNet((256, 128, args.batch_size), num_classes=args.num_classes, weight_std=True)
        model_2, start_epoch_2 = load_model(model_2, '{}/best_model_2.pth'.format(args.save))
        model_2 = nn.DataParallel(model_2)
        model_2 = model_2.cuda()
        print('加载 epoch {} 成功！'.format(start_epoch_1))
        start_epoch = start_epoch_1
    else:
        print("准备训练")
        model_1 = ConResNet((256, 256, args.batch_size), num_classes=args.num_classes, weight_std=True)
        model_1 = nn.DataParallel(model_1)
        model_1 = model_1.cuda()
        model_2 = ConResNet((256, 128, args.batch_size), num_classes=args.num_classes, weight_std=True)
        model_2 = nn.DataParallel(model_2)
        model_2 = model_2.cuda()

    optimizer_1 = optim.RMSprop(model_1.parameters(), lr=args.lr_gen, weight_decay=1e-6, momentum=0.9)
    scheduler_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min', patience=2)
    optimizer_2 = optim.RMSprop(model_2.parameters(), lr=args.lr_gen, weight_decay=1e-6, momentum=0.9)
    scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', patience=2)
    best = [0, np.inf]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for epoch in range(start_epoch, args.epochs + 1):
        iter_num_1, iter_num_2 = train(iter_num_1, iter_num_2)
        torch.cuda.empty_cache()
