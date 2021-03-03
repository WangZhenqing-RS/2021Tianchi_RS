# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""

import torch.utils.data as D
from torchvision import transforms as T
import gdal
import random
import numpy as np
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

#  读取图像像素矩阵
#  fileName 图像路径
def imgread(fileName, addNDVI=False):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    # 如果是image的话,因为label是单通道
    if(len(data.shape) == 3):
        # 添加归一化植被指数NDVI特征
        if(addNDVI):
            nir, r = data[3], data[0]
            ndvi = (nir - r) / (nir + r + 0.00001) * 1.0
            # 和其他波段保持统一,归到0-255,后面的totensor会/255统一归一化
            # 统计了所有训练集ndvi的值，最小值为0，最大值很大但是数目很少，所以我们取了98%处的25
            ndvi = (ndvi - 0) / (25 - 0) * 255
            ndvi = np.clip(ndvi, 0, 255)
            data_add_ndvi = np.zeros((5, 256, 256), np.uint8)
            data_add_ndvi[0:4] = data
            data_add_ndvi[4] = np.uint8(ndvi)
            data = data_add_ndvi
        # (C,H,W)->(H,W,C)
        data = data.swapaxes(1, 0).swapaxes(1, 2)
    return data

# 线性拉伸
def truncated_linear_stretch(image, truncated_value, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray = np.clip(gray, min_out, max_out)
        gray = np.uint8(gray)
        return gray
    
    image_stretch = []
    for i in range(image.shape[2]):
        # 只拉伸RGB
        if(i<3):
            gray = gray_process(image[:,:,i])
        else:
            gray = image[:,:,i]
        image_stretch.append(gray)
    image_stretch = np.array(image_stretch)
    image_stretch = image_stretch.swapaxes(1, 0).swapaxes(1, 2)
    return image_stretch

#  随机数据增强
#  image 图像
#  label 标签
def DataAugmentation(image, label, mode):
    if(mode == "train"):
        hor = random.choice([True, False])
        if(hor):
            #  图像水平翻转
            image = np.flip(image, axis = 1)
            label = np.flip(label, axis = 1)
        ver = random.choice([True, False])
        if(ver):
            #  图像垂直翻转
            image = np.flip(image, axis = 0)
            label = np.flip(label, axis = 0)
        stretch = random.choice([True, False])
        if(stretch):
            image = truncated_linear_stretch(image, 0.5)
    if(mode == "val"):
        stretch = random.choice([0.8, 1, 2])
    # if(stretch == 'yes'):
        # 0.5%线性拉伸
        image = truncated_linear_stretch(image, stretch)
    return image, label

#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集Iou
def cal_val_iou(model, loader):
    val_iou = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output = output.argmax(1)
        iou = cal_iou(output, target)
        val_iou.append(iou)
    return val_iou

# 计算IoU
def cal_iou(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        overlap = (p*t).sum()
        #  0.0001防止除零
        iou = 2*overlap/(uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode, addNDVI = True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.addNDVI = addNDVI
        self.len = len(image_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        image = imgread(self.image_paths[index], self.addNDVI)
        if self.mode == "train":
            label = imgread(self.label_paths[index], self.addNDVI) - 1
            image, label = DataAugmentation(image, label, self.mode)
            #  传入一个内存连续的array对象,pytorch要求传入的numpy的array对象必须是内存连续
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "val":
            label = imgread(self.label_paths[index], self.addNDVI) - 1
            # 常规来讲,验证集不需要数据增强,但是这次数据测试集和训练集不同域,为了模拟不同域,验证集也进行数据增强
            image, label = DataAugmentation(image, label, self.mode)
            image_array = np.ascontiguousarray(image)
            return self.as_tensor(image_array), label.astype(np.int64)
        elif self.mode == "test":   
            image_stretch = truncated_linear_stretch(image, 0.5)
            image_ndvi = imgread(self.image_paths[index], True)
            nir, r = image_ndvi[:,:,3], image_ndvi[:,:,0]
            ndvi = (nir - r) / (nir + r + 0.00001) * 1.0
            return self.as_tensor(image), self.as_tensor(image_stretch), self.image_paths[index], ndvi
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_paths, label_paths, mode, addNDVI, batch_size, 
                   shuffle, num_workers):
    dataset = OurDataset(image_paths, label_paths, mode, addNDVI)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def split_train_val_old(image_paths, label_paths, val_index=0, upsample = False):
    # 分隔训练集和验证集
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
    for i in range(len(image_paths)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == val_index:
            val_image_paths.append(image_paths[i])
            val_label_paths.append(label_paths[i])
        else:
            train_image_paths.append(image_paths[i])
            train_label_paths.append(label_paths[i])
    print("Number of train images: ", len(train_image_paths))
    print("Number of val images: ", len(val_image_paths))
    # 训练集上采样
    if(upsample):
        train_label_paths_upsample = train_label_paths.copy()
        train_image_paths_upsample = train_image_paths.copy()
        for i, train_label_path in enumerate(train_label_paths):
            label = imgread(train_label_path)
            label = np.unique(label)
            #print(i,len(train_label_paths),label)
            upsample_num = 2
            # 若包含少类，上采样3份,38和10因为得分太低，采取放弃策略；4因为几乎每张影像都有，采取放弃策略
            if ((5 in label) or
                (6 in label) or
                (7 in label)):
                for up in range(upsample_num):
                    train_label_paths_upsample.append(train_label_path)
                    train_image_paths_upsample.append(train_image_paths[i])
        train_label_paths = train_label_paths_upsample
        train_image_paths = train_image_paths_upsample
        print("Number of train images after upsample: ", len(train_image_paths))
        print("Number of val images after upsample: ", len(val_image_paths))  
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths

def split_train_val(image_paths, label_paths, val_index=0, upsample = True):
    # 分隔训练集和验证集
    train_image_paths, train_label_paths = image_paths, label_paths
    val_image_paths, val_label_paths = image_paths, label_paths

    print("Number of train images: ", len(train_image_paths))
    print("Number of val images: ", len(val_image_paths))
    # 训练集上采样
    if(upsample):
        train_label_paths_upsample = train_label_paths.copy()
        train_image_paths_upsample = train_image_paths.copy()
        for i, train_label_path in enumerate(train_label_paths):
            label = imgread(train_label_path)
            label = np.unique(label)
            #print(i,len(train_label_paths),label)
            upsample_num = 2
            # 若包含少类，上采样2份,38和10因为得分太低，采取放弃策略；4因为几乎每张影像都有，采取放弃策略
            if ((5 in label) or
                (6 in label) or
                (7 in label)):
                for up in range(upsample_num):
                    train_label_paths_upsample.append(train_label_path)
                    train_image_paths_upsample.append(train_image_paths[i])
        train_label_paths = train_label_paths_upsample
        train_image_paths = train_image_paths_upsample
        print("Number of train images after upsample: ", len(train_image_paths))
        print("Number of val images after upsample: ", len(val_image_paths))  
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths
# import glob
# dataset = OurDataset(
#     glob.glob(r'E:\WangZhenQing\TianChi\tcdata\suichang_round1_train_210120\*.tif'),
#     glob.glob(r'E:\WangZhenQing\TianChi\tcdata\suichang_round1_train_210120\*.png'),
#     "train",
#     True
# )
# image, label = dataset[1]
# print(image.shape, label.shape)
# print(image[0:5,0:5])
