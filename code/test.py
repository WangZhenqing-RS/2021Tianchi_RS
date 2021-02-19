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

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

def test(model_path, output_dir, test_loader, addNDVI):
    in_channels = 4
    if(addNDVI):
        in_channels += 1
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b7",
                             encoder_weights="imagenet",
                             in_channels=in_channels,
                             classes=10,
                             )
    model.to(DEVICE);
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for image, image_stretch, image_path in test_loader:
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
    model_path = "../user_data/model_data/unetpp_effi7_SoftCE_Lovasz.pth"
    output_dir = '../prediction_result'
    test_image_paths = glob.glob('../tcdata/suichang_round1_test_partA_210120/*.tif')
    addNDVI = True
    batch_size = 16
    num_workers = 8
    test_loader = get_dataloader(test_image_paths, None, "test", addNDVI, batch_size, False, 8)
    addNDVI = False
    test(model_path, output_dir, test_loader, addNDVI)