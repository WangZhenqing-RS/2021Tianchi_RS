# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:22:41 2021

@author: DELL
"""


import gdal
import numpy as np
import matplotlib.pyplot as plt
import glob

#  初始化每个类的数目
Cultivatedland_num = 0
Woodland_num = 0
Grass_num = 0
Road_num = 0
UrbanConstructionLand_num = 0
RuralConstructionLand_num = 0
IndustrialLand_num = 0
Structure_num = 0
Water_num = 0
NakedLand_num = 0

label_paths = glob.glob(r'E:\WangZhenQing\TianChi\tcdata\suichang_round1_train_210120\*.png')

for label_path in label_paths:
    label = gdal.Open(label_path).ReadAsArray(0, 0, 256, 256)
    Cultivatedland_num += np.sum(label == 1)
    Woodland_num += np.sum(label == 2)
    Grass_num += np.sum(label == 3)
    Road_num += np.sum(label == 4)
    UrbanConstructionLand_num += np.sum(label == 5)
    RuralConstructionLand_num += np.sum(label == 6)
    IndustrialLand_num += np.sum(label == 7)
    Structure_num += np.sum(label == 8)
    Water_num += np.sum(label == 9)
    NakedLand_num += np.sum(label == 10)

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

classes = ('耕地', '林地', '草地', '道路', '城镇建设用地', '农村建设用地', '工业用地', '构筑物', '水域', '裸地')
numbers = [Cultivatedland_num, Woodland_num, Grass_num, Road_num, UrbanConstructionLand_num, RuralConstructionLand_num, IndustrialLand_num, Structure_num, Water_num, NakedLand_num]

plt.barh(classes, numbers)
plt.title('地物要素类别像素数目')
plt.savefig("地物要素类别像素数目图3.png", dpi = 300, bbox_inches="tight")
plt.show()