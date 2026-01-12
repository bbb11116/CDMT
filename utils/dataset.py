import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json


BIPED_mean = [114.510, 114.451,117.230,137.86]



class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 arg=None
                 ):
        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args = arg
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")
        print(f"{test_data} 数据集加载完成，共 {len(self)} 个样本")

    def _build_index(self):
        if self.test_data == "CLASSIC":
            # 获取所有图像文件
            image_files = [
                f for f in os.listdir(self.data_root)
                if os.path.isfile(os.path.join(self.data_root, f))
                   and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            print(f"CLASSIC 模式: 在 {self.data_root} 中找到 {len(image_files)} 张图像")
            return image_files
        else:
            # 图像和标签路径位于列表文件中
            if not self.test_list:
                raise ValueError(f"数据集 {self.test_data} 未提供测试列表")

            list_name = os.path.join(self.data_root, self.test_list)
            sample_indices = []

            with open(list_name) as f:
                files = json.load(f)
            for pair in files:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (os.path.join(self.data_root, tmp_img),
                     os.path.join(self.data_root, tmp_gt),))

            return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # 获取数据样本
        if self.test_data == "CLASSIC":
            # 直接构建完整路径
            image_path = os.path.join(self.data_root, self.data_index[idx])

            # 路径验证
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"路径不存在: {image_path}")
            if not os.path.isfile(image_path):
                raise IsADirectoryError(f"路径指向目录而不是文件: {image_path}")

            img_name = os.path.basename(image_path)
            file_name = os.path.splitext(img_name)[0] + ".png"
            label_path = None
        else:
            image_path = self.data_index[idx][0]
            label_path = self.data_index[idx][1]
            img_name = os.path.basename(image_path)
            file_name = os.path.splitext(img_name)[0] + ".png"

            # 验证路径
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像路径不存在: {image_path}")
            if label_path and not os.path.exists(label_path):
                print(f"警告: 标签路径不存在 - {label_path}")
                label_path = None

        # 加载图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(
                f"无法读取图像: {image_path}\n"
                "原因: 1) 文件不存在 2) 文件损坏 3) 不支持的格式 4) 权限不足"
            )

        # 加载标签 (仅适用于非CLASSIC模式)
        if self.test_data != "CLASSIC" and label_path:
            label = cv2.imread(label_path, cv2.IMREAD_COLOR)
            if label is None:
                print(f"警告: 无法读取标签图像 - {label_path}")
                label = None
        else:
            label = None

        # 记录原始尺寸
        im_shape = [image.shape[0], image.shape[1]]

        # 应用变换
        image, label = self.transform(img=image, gt=label)

        return dict(
            images=image,
            labels=label,
            file_names=file_name,
            image_shape=im_shape
        )

    def transform(self, img, gt):
        # CLASSIC 模式处理
        if self.test_data == "CLASSIC":
            # 静默调整尺寸（不打印）
            if img.shape[:2] != (self.img_height, self.img_width):
                img = cv2.resize(img, (self.img_width, self.img_height))
            gt = None

        # 确保最小尺寸为512x512
        elif img.shape[0] < 512 or img.shape[1] < 512:
            img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height))
            if gt is not None:
                gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height))

        # 确保尺寸能被16整除
        elif img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
            img_width = ((img.shape[1] // 8) + 1) * 8
            img_height = ((img.shape[0] // 8) + 1) * 8
            img = cv2.resize(img, (img_width, img_height))
            if gt is not None:
                gt = cv2.resize(gt, (img_width, img_height))

        # 转换图像格式
        img = np.array(img, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        # CLASSIC 模式的特殊处理
        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[1], img.shape[2]))  # 创建空标签
            gt = torch.from_numpy(np.array([gt])).float()
        elif gt is not None:
            # 处理有效标签
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]  # 取单通道
            gt /= 255.0  # 归一化
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            # 创建空标签
            gt = torch.zeros(1, img.shape[1], img.shape[2])

        return img, gt


class BipedDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg
        self.data_index = self._build_index()

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []
        if self.arg.train_data.lower() == 'biped':

            images_path = os.path.join(data_root,
                                       'imgs',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)
            labels_path = os.path.join(data_root,
                                       'edges',
                                       self.train_mode,
                                       self.dataset_type,
                                       self.data_type)

            for directory_name in os.listdir(images_path):
                image_directories = os.path.join(images_path, directory_name)
                for file_name_ext in os.listdir(image_directories):
                    file_name = os.path.splitext(file_name_ext)[0]
                    sample_indices.append(
                        (os.path.join(images_path, directory_name, file_name + '.jpg'),
                         os.path.join(labels_path, directory_name, file_name + '.png'),)
                    )
        else:
            file_path = os.path.join(data_root, self.arg.train_list)
            if self.arg.train_data.lower() == 'bsds':

                with open(file_path, 'r') as f:
                    files = f.readlines()
                files = [line.strip() for line in files]

                pairs = [line.split() for line in files]
                for pair in pairs:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))
            else:
                with open(file_path) as f:
                    files = json.load(f)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(data_root, tmp_img),
                         os.path.join(data_root, tmp_gt),))



        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path ,circles = self.data_index[idx]
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label )

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.  # for LDC input and BDCN

        img = np.array(img, dtype=np.float32)

        # 删减此处是否会有影响？2024.06.27
        # img -= self.mean_bgr

        #  400 for BIPEd and 352 for BSDS check with 384
        crop_size = self.img_height if self.img_height == self.img_width else None  # 448# MDBD=480 BIPED=480/400 BSDS=352

        # for BSDS 352/BRIND
        # if i_w > crop_size and i_h > crop_size:  # later 400, before crop_size
        #     i = random.randint(0, i_h - crop_size)
        #     j = random.randint(0, i_w - crop_size)
        #     img = img[i:i + crop_size, j:j + crop_size]
        #     gt = gt[i:i + crop_size, j:j + crop_size]
        #
        # # # for BIPED/MDBD
        # # if i_w> 420 and i_h>420: #before np.random.random() > 0.4
        # #     h,w = gt.shape
        # #     if np.random.random() > 0.4: #before i_w> 500 and i_h>500:
        # #
        # #         LR_img_size = crop_size #l BIPED=256, 240 200 # MDBD= 352 BSDS= 176
        # #         i = random.randint(0, h - LR_img_size)
        # #         j = random.randint(0, w - LR_img_size)
        # #         # if img.
        # #         img = img[i:i + LR_img_size , j:j + LR_img_size ]
        # #         gt = gt[i:i + LR_img_size , j:j + LR_img_size ]
        # #     else:
        # #         LR_img_size = 300#208  # l BIPED=208-352, # MDBD= 352-480- BSDS= 176-320
        # #         i = random.randint(0, h - LR_img_size)
        # #         j = random.randint(0, w - LR_img_size)
        # #         # if img.
        # #         img = img[i:i + LR_img_size, j:j + LR_img_size]
        # #         gt = gt[i:i + LR_img_size, j:j + LR_img_size]
        # #         img = cv2.resize(img, dsize=(crop_size, crop_size), )
        # #         gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        #
        # else:
        #     # New addidings
        #     img = cv2.resize(img, dsize=(crop_size, crop_size))
        #     gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        # # BRIND
        # gt[gt > 0.1] +=0.2#0.4
        # gt = np.clip(gt, 0., 1.)
        # for BIPED
        gt[gt > 0.2] += 0.6  # 0.5 for BIPED
        gt = np.clip(gt, 0., 1.)  # BIPED
        # # for MDBD
        # gt[gt > 0.3] +=0.7#0.4
        # gt = np.clip(gt, 0., 1.)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()

        # _, i_h, i_w = img.shape
        # if i_h != self.img_height or i_w != self.img_width:
        #     img = self.transforms(img)
        #     gt = self.transforms(gt)
        return img, gt