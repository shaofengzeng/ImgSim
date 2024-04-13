#-*-coding:utf8-*-
'''
将voc图片数据分为训练和测试集
并将图片路径分别保存至test.txt和train.txt
'''


import random
import os

data_dir = r"D:\data\VOCdevkit\VOC2012\JPEGImages"
im_names = os.listdir(data_dir)
image_list = []
for n in im_names:
    full_path = os.path.join(data_dir,n)
    image_list.append(full_path)


#split train test
test_idx = random.choices(range(len(image_list)),k=int(len(image_list)*0.3))
train_idx = [i for i in range(len(image_list)) if i not in test_idx]

with open(r"..\datasets\train.txt","w") as fout:
    for _idx in train_idx:
        fout.write(image_list[_idx]+"\n")

with open(r"..\datasets\test.txt","w") as fout:
    for _idx in test_idx:
        fout.write(image_list[_idx]+"\n")