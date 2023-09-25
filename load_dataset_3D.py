import itertools
import os
import cv2
import tifffile
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch
import numpy as np
import SimpleITK as sitk
import config

args = config.args


def equal(im, flg_max):
    flg_min = 0.01
    mi = im.min()
    mx = im.max()
    imax = flg_max * mx + (1 - flg_max) * mi
    imin = flg_min * mx + (1 - flg_min) * mi

    im[im > imax] = imax
    im[im < imin] = imin
    return (im - np.min(im)) / (np.max(im) - np.min(im))


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list




class BasicDataset_lsfm(Dataset):
    def __init__(self, path, name_label, a):
        self.path = path
        self.img = []
        self.mask = []
        self.res=[]
        self.a = a
        directoryname = load_file_name_list(os.path.join(path, name_label))
        self.num = len(directoryname)

        for d in directoryname:
            name = 'mask_' + args.name
            m = d.replace('tif', 'ome.nii')
            m = m.replace('data', name)
            self.mask.append(os.path.join(path, m))
            num = (d[-8:-4])
            if (self.a == 'c' and num == '0511'):
                num_res = str(511)
            elif (self.a == 'a' and num == '0319'):
                num_res = str(319)
            else:
                num_res = str(int(d[-8:-4]) + 1)
            num_res=num_res.zfill((4))
            res=m[:-12]+num_res+m[-8:]

            self.img.append(os.path.join(path, d))
            self.res.append(os.path.join(path, res))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        tmp = []
        img_name = self.img[i]
        mask_name = self.mask[i]
        res_name = self.res[i]
        data_img = tifffile.imread(img_name)

        if self.a == 'c':
            new_x, new_y = 448, 320  # c
        if self.a == 'a':
            new_x, new_y = 512, 448  # a
        dmin = data_img.min()
        dmax = data_img.max()
        data_img = (data_img - dmin) / (dmax - dmin + 1)
        data_img_new = []
        data_img_new.append(data_img)
        data_img_new = np.array(data_img_new)
        data_img = data_img_new.astype(float)
        data_img = torch.from_numpy(data_img)
        if (i < self.num):
            data_mask = sitk.ReadImage(mask_name)
            data_mask = sitk.GetArrayFromImage(data_mask)
            data_mask = 1 * np.asarray(data_mask).astype('uint16')
            cv2.imwrite('temp_m/mask.tiff', data_mask)
            data_mask1 = np.zeros_like(data_mask)
            data_mask1[data_mask == 2] = 1
            data_mask1 = cv2.resize(data_mask1, (new_x, new_y))
            try:
                res_mask = sitk.ReadImage(res_name)
            except:
                print(res_name,img_name,mask_name)
            res_mask = sitk.GetArrayFromImage(res_mask)
            res_mask1 = np.zeros_like(res_mask)
            res_mask1[res_mask == 2] = 1
            res_mask1 = abs(res_mask1 - data_mask1)
            res_mask1 = 1 * np.asarray(res_mask1).astype('uint16')
            res_mask_new = []
            res_mask_new.append(res_mask1)
            res_mask_new = np.array(res_mask_new)
            res_mask_new = res_mask_new / 1.0
            data_mask_new = []
            data_mask_new.append(data_mask1)
            # data_mask_new.append(data_mask2)
            # data_mask_new.append(data_mask3)
            # data_mask_new.append(data_mask4)
            data_mask_new = np.array(data_mask_new)
        else:
            #            data_mask_new = [0]
            data_mask_new = np.zeros((1, new_y, new_x))
            res_mask_new = np.zeros((1, new_y, new_x))

        data_mask_new = data_mask_new.astype(float)
        data_mask_new = torch.from_numpy(data_mask_new)

        return {"img": data_img, "cate": data_mask_new, "mask_name": mask_name,"res":res_mask_new,"res_name":res_name}
class BasicDataset_res_fvb(Dataset):
    def __init__(self, path, name_label, a):
        self.path = path
        self.img = []
        self.mask = []
        self.res=[]
        self.a = a
        directoryname = load_file_name_list(os.path.join(path, name_label))
        self.num = len(directoryname)
        for d in directoryname:
            # name = 'mask_all_' + args.name
            name = 'mask_all'
            m = d.replace('tif', 'nii')
            m = m.replace('data', name)
            # m = m.replace('_647_m', '_m')
            #            m = m.replace('_r6_m', '_m')
            num = (d[-8:-4])
            if (self.a == 'c' and num == '0447'):
                num_res = 447
            elif (self.a == 'a' and num == '0223'):
                num_res = 223
            else:
                num_res = str(int(d[-8:-4]) + 1)
            res = m.replace(num, str(num_res).zfill(4))

            self.mask.append(os.path.join(path, m))
            self.img.append(os.path.join(path, d))
            self.res.append(os.path.join(path, res))



    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        tmp = []
        img_name = self.img[i]
        mask_name = self.mask[i]
        res_name= self.res[i]
        #        print(img_name, mask_name)
        scale = 0.5
        data_img = tifffile.imread(img_name)


        #        new_x, new_y = 224, 224

        if self.a == 'c':
           # new_x, new_y = 448, 320  # c
           new_x, new_y = 288, 224
           data_img = np.rot90(data_img, 2)
        #            new_x, new_y = 256, 256
        if self.a == 'a':
          #  new_x, new_y = 512, 448  # a
           new_x, new_y = 448, 288
        #            new_x, new_y = 512, 256
        #        data_img = cv2.resize(data_img, (new_x, new_y))
        #        cv2.imwrite('temp_m/data.tiff', data_img)
        dmin = data_img.min()
        dmax = data_img.max()
        data_img = (data_img - dmin) / (dmax - dmin + 1)
        data_img_new = []
        data_img_new.append(data_img)

        data_img_new = np.array(data_img_new)
        data_img = data_img_new.astype(float)
        data_img = torch.from_numpy(data_img)

        if (i < self.num):
            data_mask = sitk.ReadImage(mask_name)
            data_mask = sitk.GetArrayFromImage(data_mask)
            if(self.a=='c'):
                data_mask = np.rot90(data_mask, 2)
            data_mask = 1 * np.asarray(data_mask).astype('uint16')

            res_mask = sitk.ReadImage(res_name)
            res_mask = sitk.GetArrayFromImage(res_mask)
            if (self.a == 'c'):
                res_mask = np.rot90(res_mask, 2)
            res_mask = 1 * np.asarray(res_mask).astype('uint16')


            #            print(mask_name, data_mask.dtype)
            #            data_mask = cv2.resize(data_mask, (new_x, new_y))

            data_mask1 = np.zeros_like(data_mask)
            res_mask1 = np.zeros_like(res_mask)
            # data_mask1[data_mask == 1] = 1
            # data_mask1 = cv2.resize(data_mask1, (new_x, new_y))

            # CTX
            # data_mask1[data_mask == 14] = 1
            # data_mask1[data_mask == 34] = 1
            # data_mask1[data_mask == 16] = 1
            # data_mask1[data_mask == 36] = 1
            # res_mask1[res_mask == 14] = 1
            # res_mask1[res_mask == 34] = 1
            # res_mask1[res_mask == 16] = 1
            # res_mask1[res_mask == 36] = 1

            # HPF
            data_mask1[data_mask == 1] = 1
            data_mask1[data_mask == 21] = 1
            res_mask1[res_mask == 1] = 1
            res_mask1[res_mask == 21] = 1

            # CP
            # data_mask1[data_mask == 3] = 1
            # data_mask1[data_mask == 23] = 1
            # res_mask1[data_mask == 3] = 1
            # res_mask1[data_mask == 23] = 1

            # BS
            # data_mask1[data_mask == 17] = 1
            # data_mask1[data_mask == 32] = 1
            # data_mask1[data_mask == 12] = 1
            # data_mask1[data_mask == 29] = 1
            # data_mask1[data_mask == 9] = 1
            # data_mask1[data_mask == 38] = 1
            # data_mask1[data_mask == 18] = 1
            # data_mask1[data_mask == 33] = 1
            # data_mask1[data_mask == 13] = 1
            # data_mask1[data_mask == 7] = 1
            # data_mask1[data_mask == 27] = 1
            # data_mask1[data_mask == 31] = 1
            # data_mask1[data_mask == 11] = 1
            # data_mask1[data_mask == 38] = 1
            # data_mask1[data_mask == 18] = 1
            #
            # res_mask1[data_mask == 17] = 1
            # res_mask1[data_mask == 32] = 1
            # res_mask1[data_mask == 12] = 1
            # res_mask1[data_mask == 29] = 1
            # res_mask1[data_mask == 9] = 1
            # res_mask1[data_mask == 38] = 1
            # res_mask1[data_mask == 18] = 1
            # res_mask1[data_mask == 33] = 1
            # res_mask1[data_mask == 13] = 1
            # res_mask1[data_mask == 7] = 1
            # res_mask1[data_mask == 27] = 1
            # res_mask1[data_mask == 31] = 1
            # res_mask1[data_mask == 11] = 1
            # res_mask1[data_mask == 38] = 1
            # res_mask1[data_mask == 18] = 1

            # CB
            # data_mask1[data_mask == 28] = 1
            # data_mask1[data_mask == 8] = 1
            # res_mask1[res_mask == 28] = 1
            # res_mask1[res_mask == 8] = 1


            res_mask1=abs(res_mask1-data_mask1)
            res_mask1 = 1 * np.asarray(res_mask1).astype('uint16')
            # res_mask1=np.zeros_like(data_mask1)

        #    data_mask2 = np.zeros_like(data_mask)

            # data_mask2[data_mask == 2] = 1
            # data_mask2 = cv2.resize(data_mask2, (new_x, new_y))
            #            if 'allen' in mask_name:
            #                data_mask2[data_mask2 > 0] = 0
            #                data_mask2[data_mask == 3] = 1
            #            data_mask3 = np.zeros_like(data_mask)
            #            data_mask3[data_mask == 3] = 1
            #            data_mask3 = cv2.resize(data_mask3, (new_x, new_y))
            #            data_mask4 = np.zeros_like(data_mask)
            #            data_mask4[data_mask == 4] = 1
            #            data_mask4 = cv2.resize(data_mask4, (new_x, new_y))
            #             cv2.imwrite('temp_m/mask1.tiff', data_mask1)
            # cv2.imwrite('temp_m/mask2.tiff', data_mask2)
            #            cv2.imwrite('temp_m/mask3.tiff', data_mask3)
            #            cv2.imwrite('temp_m/mask4.tiff', data_mask4)
            #            print(mask_name, data_mask1.max(), data_mask1.min())
            data_mask_new = []
            data_mask_new.append(data_mask1)
            # data_mask_new.append(data_mask2)
            #            data_mask_new.append(data_mask3)
            #            data_mask_new.append(data_mask4)
            data_mask_new = np.array(data_mask_new)
            res_mask_new = []
            res_mask_new.append(res_mask1)

            res_mask_new = np.array(res_mask_new)
            res_mask_new = res_mask_new / 1.0

        else:
            #            data_mask_new = [0]
            data_mask_new = np.zeros((1, new_y, new_x))
            res_mask_new = np.zeros((1, new_y, new_x))

        data_mask_new = data_mask_new.astype(float)
        data_mask_new = torch.from_numpy(data_mask_new)


        return {"img": data_img, "cate": data_mask_new, "mask_name": mask_name,"res":res_mask_new,"res_name":res_name}
class BasicDataset_res_test(Dataset):
    def __init__(self, path, name,a):
        self.path = path
        self.img = []
        self.mask = []
        self.res=[ ]
        self.a = a

        # 测试新的八个脑子
        directoryname = load_file_name_list(os.path.join(path, name))
        for d in directoryname:
            name = 'mask'
            m = d.replace('tif', 'nii')
            m = m.replace('data', name)
            self.mask.append(os.path.join(path, m))
            num = (d[-8:-4])
            if (self.a == 'c' and num == '0511'):
                num_res = str(511)
            elif (self.a == 'a' and num == '0319'):
                num_res = str(319)
            else:
                num_res = str(int(d[-8:-4]) + 1)
            num_res = num_res.zfill((4))
            res = m[:-12] + num_res + m[-8:]
            self.img.append(os.path.join(path, d))
            self.res.append(os.path.join(path, res))
    def img_pretreatment(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.new_W, self.new_H))
        return img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img_name = self.img[i]
        mask_name = self.mask[i]
        data_img = tifffile.imread(img_name)
        dmin = data_img.min()
        dmax = data_img.max()
        if dmin != dmax:
            data_img = (data_img - dmin) / (dmax - dmin)
        data_img = data_img.astype(float)
        data_img = data_img.reshape(1, data_img.shape[0], data_img.shape[1])
        data_img = torch.from_numpy(data_img)
        res_img=np.zeros_like(data_img)
        dmin = res_img.min()
        dmax = res_img.max()
        res_img = (res_img - dmin) / (dmax - dmin + 1)
        res_img_new = []
        res_img_new.append(res_img)

        return {"img": data_img, "mask_name": mask_name,"res":res_img}
class BasicDataset_res_test_fvb(Dataset):
    def __init__(self, path, name,a):
        self.path = path
        self.img = []
        self.mask = []
        self.res=[ ]
        self.a = a

        # 测试新的八个脑子
        directoryname = load_file_name_list(os.path.join(path, name))
        for d in directoryname:
            name = 'mask_all'
            m = d.replace('tif', 'nii')
            m = m.replace('data', name)
            self.mask.append(os.path.join(path, m))
            num = (d[-8:-4])
            if (self.a == 'c' and num == '0447'):
                num_res = 447
            elif (self.a == 'a' and num == '0223'):
                num_res = 223
            else:
                num_res = str(int(d[-8:-4]) + 1)
            res = m.replace(num, str(num_res).zfill(4))
            self.img.append(os.path.join(path, d))
            self.res.append(os.path.join(path, res))
    def img_pretreatment(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.new_W, self.new_H))
        return img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img_name = self.img[i]
        mask_name = self.mask[i]
        data_img = tifffile.imread(img_name)
        if(self.a=='c'):
            data_img = np.rot90(data_img, 2)

        dmin = data_img.min()
        dmax = data_img.max()
        if dmin != dmax:
            data_img = (data_img - dmin) / (dmax - dmin)
        data_img = data_img.astype(float)
        data_img = data_img.reshape(1, data_img.shape[0], data_img.shape[1])
        data_img = torch.from_numpy(data_img)

        res_img=np.zeros_like(data_img)  #测试时res=0
        dmin = res_img.min()
        dmax = res_img.max()
        res_img = (res_img - dmin) / (dmax - dmin + 1)

        res_img_new = []
        res_img_new.append(res_img)
        return {"img": data_img, "mask_name": mask_name,"res":res_img}
class BasicDataset_lsfm_test(Dataset):
    def __init__(self, path, name_label, a):
        self.path = path
        self.img = []
        self.mask = []
        self.res=[]
        self.a = a
        directoryname = load_file_name_list(os.path.join(path, name_label))
        self.num = len(directoryname)

        for d in directoryname:
            name = 'mask_' + args.name
            m = d.replace('tif', 'ome.nii')
            m = m.replace('data', name)
            self.mask.append(os.path.join(path, m))
            num = (d[-8:-4])
            if (self.a == 'c' and num == '0511'):
                num_res = str(511)
            elif (self.a == 'a' and num == '0319'):
                num_res = str(319)
            else:
                num_res = str(int(d[-8:-4]) + 1)
            num_res=num_res.zfill((4))
            res=m[:-12]+num_res+m[-8:]

            self.img.append(os.path.join(path, d))
            self.res.append(os.path.join(path, res))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img_name = self.img[i]
        mask_name = self.mask[i]
        res_name = self.res[i]
        #        print(img_name, mask_name)
        data_img = tifffile.imread(img_name)
        dmin = data_img.min()
        dmax = data_img.max()
        if dmin != dmax:
            data_img = (data_img - dmin) / (dmax - dmin)

        a = []
        data_img = data_img.astype(float)

        data_img = data_img.reshape(1, data_img.shape[0], data_img.shape[1])

        data_img = torch.from_numpy(data_img)

        # res_img = tifffile.imread(res_name)
        res_img = np.zeros_like(data_img)
        dmin = res_img.min()
        dmax = res_img.max()
        res_img = (res_img - dmin) / (dmax - dmin + 1)

        res_img_new = []
        res_img_new.append(res_img)
        # res_img_new = np.array(res_img_new)
        # res_img = res_img_new.astype(float)
        # res_img = torch.from_numpy(res_img)
        # res_img = abs(res_img - data_img)

        return {"img": data_img, "mask_name": mask_name, "res": res_img}
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        print(len(self.secondary_indices),self.secondary_batch_size)

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)