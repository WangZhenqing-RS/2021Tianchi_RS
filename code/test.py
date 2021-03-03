# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:51:44 2021

@author: DELL
"""
import glob
from dataProcess import get_dataloader
import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
from torch.optim.swa_utils import AveragedModel
import time

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

def test(model_path_effi7, model_path_resnest, output_dir, test_loader, addNDVI):
    in_channels = 4
    if(addNDVI):
        in_channels += 1
    model_resnest = smp.UnetPlusPlus(
        encoder_name="timm-resnest101e",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=10,
        )
    model_effi7 = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=10,   
        )
    # 如果模型是SWA
    if("swa" in model_path_resnest):
        model_resnest = AveragedModel(model_resnest)
    if("swa" in model_path_effi7):
        model_effi7 = AveragedModel(model_effi7)
    model_resnest.to(DEVICE);
    model_resnest.load_state_dict(torch.load(model_path_resnest))
    model_resnest.eval()
    model_effi7.to(DEVICE);
    model_effi7.load_state_dict(torch.load(model_path_effi7))
    model_effi7.eval()
    for image, image_stretch, image_path, ndvi in test_loader:
        with torch.no_grad():
            # image.shape: 16,4,256,256
            image_flip2 = torch.flip(image,[2])
            image_flip2 = image_flip2.cuda()
            image_flip3 = torch.flip(image,[3])
            image_flip3 = image_flip3.cuda()
            image = image.cuda()
            image_stretch = image_stretch.cuda()
            
            output1 = model_resnest(image).cpu().data.numpy()
            output2 = model_resnest(image_stretch).cpu().data.numpy()
            output3 = model_effi7(image).cpu().data.numpy()
            output4 = model_effi7(image_stretch).cpu().data.numpy()
            
            output5 = torch.flip(model_resnest(image_flip2),[2]).cpu().data.numpy()
            output6 = torch.flip(model_effi7(image_flip2),[2]).cpu().data.numpy()
            output7 = torch.flip(model_resnest(image_flip3),[3]).cpu().data.numpy()
            output8 = torch.flip(model_effi7(image_flip3),[3]).cpu().data.numpy()
            
        output = (output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8) / 8.0
        # output.shape: 16,10,256,256
        for i in range(output.shape[0]):
            pred = output[i]
            # for low_ndvi in range(3,8):
            #     pred[low_ndvi][ndvi[i]>35] = 0
            # for high_ndvi in range(3):
            #     pred[high_ndvi][ndvi[i]<0.02] = 0
            pred = np.argmax(pred, axis = 0) + 1
            pred = np.uint8(pred)
            save_path = os.path.join(output_dir, image_path[i][-10:].replace('.tif', '.png'))
            print(save_path)
            cv2.imwrite(save_path, pred)
        
def test_1(model_path, output_dir, test_loader, addNDVI):
    in_channels = 4
    if(addNDVI):
        in_channels += 1
    # model = smp.UnetPlusPlus(
    #         encoder_name="resnet101",
    #         encoder_weights="imagenet",
    #         in_channels=4,
    #         classes=10,
    # )
    model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=4,
            classes=10,
    )
    # 如果模型是SWA
    if("swa" in model_path):
        model = AveragedModel(model)
    model.to(DEVICE);
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for image, image_stretch, image_path, ndvi in test_loader:
        with torch.no_grad():
            image = image.cuda()
            image_stretch = image_stretch.cuda()
            output1 = model(image).cpu().data.numpy()
            output2 = model(image_stretch).cpu().data.numpy()
        output = (output1 + output2) / 2.0
        for i in range(output.shape[0]):
            pred = output[i]
            pred = np.argmax(pred, axis = 0) + 1
            pred = np.uint8(pred)
            save_path = os.path.join(output_dir, image_path[i][-10:].replace('.tif', '.png'))
            print(save_path)
            cv2.imwrite(save_path, pred)
        
if __name__ == "__main__":
    start_time = time.time()
    model_path_effi7 = "../user_data/model_data/unetplusplus_effi7_upsample_SoftCE_dice.pth"
    model_path_resnest = "../user_data/model_data/unetplusplus_resnest_upsample_SoftCE_dice.pth"
    # model_path = "../user_data/model_data/deeplabv3_resnet_upsample_SoftCE_dice.pth"
    output_dir = '../prediction_result'  
    test_image_paths = glob.glob('../tcdata/suichang_round1_test_partA_210120/*.tif')
    addNDVI = False
    batch_size = 16
    num_workers = 8
    test_loader = get_dataloader(test_image_paths, None, "test", addNDVI, batch_size, False, 8)
    test(model_path_effi7, model_path_resnest, output_dir, test_loader, addNDVI)
    # test_1(model_path, output_dir, test_loader, addNDVI)
    print((time.time()-start_time)/60**1)
