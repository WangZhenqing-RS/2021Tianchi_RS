# -*- coding: utf-8 -*-
"""
@author: xinyi
ref:
1.https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.6cc26423Zxyf0s&postId=169396
2.https://github.com/DLLXW/data-science-competition/tree/main/%E5%A4%A9%E6%B1%A0
3.https://github.com/JasmineRain/NAIC_AI-RS/tree/ec70861e2a7f3ba18b3cc8bad592e746145088c9
"""
import numpy as np
import torch
import warnings
import time
from dataProcess import get_dataloader, cal_val_iou, split_train_val
import segmentation_models_pytorch as smp
import glob
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
from pytorch_toolbelt import losses as L
from torch.optim.swa_utils import AveragedModel, SWALR
## 使用自动混合精度训练，在尽可能减少精度损失的情况下利用半精度浮点数加速训练
#from torch.cuda.amp import autocast, GradScaler

# 忽略警告信息
warnings.filterwarnings('ignore')
# cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
torch.backends.cudnn.enabled = True

# Tensor和Numpy都是矩阵,区别是前者可以在GPU上运行,后者只能在CPU上
# 但是Tensor和Numpy互相转化很方便
# 将模型加载到指定设备DEVICE上
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


def train(EPOCHES, BATCH_SIZE, train_image_paths, train_label_paths, 
          val_image_paths, val_label_paths, channels, optimizer_name,
          model_path, swa_model_path, addNDVI, loss):
    
    train_loader = get_dataloader(train_image_paths, train_label_paths, 
                                  "train", addNDVI, BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = get_dataloader(val_image_paths, val_label_paths, 
                                  "val", addNDVI, BATCH_SIZE, shuffle=False, num_workers=8)
    
    # 定义模型,优化器,损失函数
    model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=channels,
            classes=10,
    )
    model.to(DEVICE);
    
    # 采用SGD优化器
    if(optimizer_name == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=1e-4, weight_decay=1e-3, momentum=0.9)
    # 采用AdamM优化器
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=1e-4, weight_decay=1e-3)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
            ) 
    # 使用SWA的初始epoch,一般是EPOCHES的0.75倍
    swa_start = int(EPOCHES * 0.75)
    # 随机权重平均SWA,以几乎不增加任何成本的方式实现更好的泛化
    swa_model = AveragedModel(model).to(DEVICE)
    # SWA调整学习率
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    # loss_fn = nn.CrossEntropyLoss().to(DEVICE);
    
    if(loss == "SoftCE_dice"):
        # 损失函数采用SoftCrossEntropyLoss+DiceLoss
        # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
        DiceLoss_fn=DiceLoss(mode='multiclass')
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
    else:
        # 损失函数采用SoftCrossEntropyLoss+LovaszLoss
        # LovaszLoss是对基于子模块损失凸Lovasz扩展的mIoU损失的直接优化
        LovaszLoss_fn = LovaszLoss(mode='multiclass')
        # 软交叉熵,即使用了标签平滑的交叉熵,会增加泛化性
        SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        loss_fn = L.JointLoss(first=LovaszLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
    
    header = r'Epoch/EpochNum | TrainLoss | ValidmIoU | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.3f} | {:9.3f} | {:9.2f}'
    print(header)
    
#    # 在训练最开始之前实例化一个GradScaler对象,使用autocast才需要
#    scaler = GradScaler()

    # 记录当前验证集最优mIoU,以判定是否保存当前模型
    best_miou = 0
    train_loss_epochs, val_mIoU_epochs, lr_epochs = [], [], []
    # 开始训练
    for epoch in range(1, EPOCHES+1):
        # print("Start training the {}st epoch...".format(epoch))
        # 存储训练集每个batch的loss
        losses = []
        start_time = time.time()
        model.train()
        model.to(DEVICE);
        for batch_index, (image, target) in enumerate(train_loader):
            image, target = image.to(DEVICE), target.to(DEVICE)
            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
#            # 使用autocast半精度加速训练,前向过程(model + loss)开启autocast
#            with autocast(): #need pytorch>1.6
            # 模型推理得到输出
            output = model(image)
            # 求解该batch的loss
            loss = loss_fn(output, target)
#                scaler.scale(loss).backward()
#                scaler.step(optimizer)
#                scaler.update()
            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            losses.append(loss.item())
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # 余弦退火调整学习率
            scheduler.step()
        # 计算验证集IoU
        val_iou = cal_val_iou(model, valid_loader)
        # 输出验证集每类IoU
        # print('\t'.join(np.stack(val_iou).mean(0).round(3).astype(str)))
        # 保存当前epoch的train_loss.val_mIoU.lr_epochs
        train_loss_epochs.append(np.array(losses).mean())
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        # 输出进程
        print(raw_line.format(epoch, EPOCHES, np.array(losses).mean(), 
                              np.mean(val_iou), 
                              (time.time()-start_time)/60**1), end="")    
        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            torch.save(model.state_dict(), model_path)
            print("  valid mIoU is improved. the model is saved.")
        else:
            print("")
    # 最后更新BN层参数
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device= DEVICE)
    # 计算验证集IoU
    val_iou = cal_val_iou(model, valid_loader)
    print("swa_model'mIoU is {}".format(np.mean(val_iou)))
    torch.save(swa_model.state_dict(), swa_model_path)
    return train_loss_epochs, val_mIoU_epochs, lr_epochs

# 不加主函数这句话的话,Dataloader多线程加载数据会报错
if __name__ == '__main__':
    EPOCHES = 80
    BATCH_SIZE = 16
    image_paths = glob.glob(r'..\tcdata\suichang_round1_train_210120\*.tif')
    label_paths = glob.glob(r'..\tcdata\suichang_round1_train_210120\*.png')
    # 每5个数据的第val_index个数据为验证集
    val_index = 0
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths, 
                                                                                             label_paths, 
                                                                                             val_index)
    loss = "SoftCE_dice"
    #loss = "SoftCE_Lovasz"
    channels = 4
    addNDVI = True
    if(addNDVI):
        channels += 1
    optimizer_name = "adamw"
    model_path = "../user_data/model_data/unetpp_effi7"
    if(addNDVI):
        model_path += "_ndvi"
    model_path += "_" + loss
    swa_model_path = model_path + "_swa.pth"
    model_path += ".pth"
    train_loss_epochs, val_mIoU_epochs, lr_epochs = train(EPOCHES, 
                                                          BATCH_SIZE, 
                                                          train_image_paths, 
                                                          train_label_paths, 
                                                          val_image_paths, 
                                                          val_label_paths, 
                                                          channels, 
                                                          optimizer_name,
                                                          model_path, 
                                                          swa_model_path, 
                                                          addNDVI,
                                                          loss)
    if(True):    
        import matplotlib.pyplot as plt
        x = [i for i in range(EPOCHES)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, train_loss_epochs, label='train loss')
        ax.plot(x, val_mIoU_epochs, label='val mIoU')
        ax.set_xlabel('epoch', fontsize=15)
        ax.set_ylabel('metrics', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, lr_epochs,  label='learning rate')
        ax.set_xlabel('epoch', fontsize=15)
        ax.set_ylabel('learning rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.savefig("train curve", dpi=300)
        plt.show()