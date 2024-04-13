import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as tsf
from shapely.geometry import Polygon
from utils.homo_aug import sample_homography
from utils.photo_aug import PhotoAug
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def remove_duplicate_elements(arr):
    '''
    去除arr中冗余元素
    :param arr: np.array
    '''
    _arr = arr.tolist()
    _arr = [tuple(a) for a in _arr]
    _arr = set(_arr)
    return np.array(list(_arr))

def polygon_inter_union_ratio(p1, p2):
    '''
    多边形p1与p2之间的交并比
    :param p1: Polygon([(0,0), (1,2),...])
    :param p2: 与1相似
    :return:
    '''
    # 计算 p1 和 p2 的交叠面积
    overlap = p1.intersection(p2)
    union = p1.union(p2)
    # 计算交叠面积占 B 框的比例
    overlap_ratio = overlap.area/union.area
    return overlap_ratio

def draw_points(img, points):
    color = (0, 255, 0) # 绿色
    # 设置线条粗细，-1将使线条变得粗，这对于绘制点非常有用
    thickness = -1
    _img = img.copy()
    for p in points:
        cv2.circle(_img, (int(p[0]), int(p[1])), 1, color, thickness)
    return _img

def draw_lines(img, points):
    color = (0, 255, 0)  # 绿色
    # 设置线条粗细，-1将使线条变得粗，这对于绘制点非常有用
    thickness = -1
    _img = img.copy()
    N = len(points)
    for i in range(N):
        if i==(N-1):
            cv2.line(_img, (int(points[i][0]), int(points[i][1])), (int(points[0][0]), int(points[0][1])), (0,255,0), 1)
        else:
            cv2.line(_img,(int(points[i][0]), int(points[i][1])), (int(points[i+1][0]), int(points[i+1][1])),(0,255,0), 1)
    return _img

class SimDataSet(Dataset):
    '''
    从图像中,截取局部区域, 形成训练数据集
    '''
    def __init__(self, image_list_file, is_train=True, output_shape=(160, 120)):
        '''
        :param image_list: 图像路径
        :param image_aug: 是否进行图像增强
        :param output_shape: 输出图像的大小,若为None则不同batch图像大小不一致
        '''
        self.image_list = []
        with open(image_list_file, "r") as fin:
            for line in fin:
                self.image_list.append(line.strip())

        self.is_train = is_train
        self.output_shape = output_shape
        self.aug = PhotoAug()


    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        im_path = self.image_list[index]
        img = cv2.imread(im_path)
        im_h, im_w = img.shape[0:2]
        c_w, c_h = self.output_shape

        H = sample_homography((im_w, im_h),)
        H_inv = np.linalg.inv(H)
        homo_img = cv2.warpPerspective(img, H, (im_w, im_h))

        if im_w<(1.5*c_w) or im_h<(1.5*c_h):
            return None, None, None

        #在homo_img上随机选择一些点,作为预选图像区域的中心点
        homo_rand_x = np.random.randint(low=1+c_w//2, high=im_w-1-c_w//2, size=int((im_w*im_h)/(32*32)))
        homo_rand_y = np.random.randint(low=1+c_h//2, high=im_h-1-c_h//2, size=int((im_w*im_h)/(32*32)))
        homo_rand_points = np.stack([homo_rand_x, homo_rand_y], axis=1)

        # 将homo_rand_points映射到img中
        rand_points = cv2.perspectiveTransform(homo_rand_points.astype(np.float32).reshape(-1, 1, 2), H_inv).squeeze().astype(np.int32)

        #移除无法完整取大小为(c_w,c_h)的图像区域的中心点
        reserve = (rand_points[:,0]>(c_w//2)) & (rand_points[:,0]<(im_w-1-c_w//2))& \
                  (rand_points[:,1]>(c_h//2)) & (rand_points[:,1]<(im_h-1-c_h//2))
        homo_rand_points = homo_rand_points[reserve]
        rand_points = rand_points[reserve]

        #依据homo_rand_points, 在homo_img上取h_box=[h_min_x, h_min_y, h_max_x, h_max_y]
        h_min_x, h_min_y = homo_rand_points[:,0]-c_w//2, homo_rand_points[:,1]-c_h//2
        h_max_x, h_max_y = h_min_x+c_w, h_min_y+c_h
        assert np.all((h_min_x >= 0) & (h_min_y >= 0) & (h_max_x <= im_w) & (h_max_y <= im_h))

        #依据rand_points，在img上取box=[min_x, min_y, max_x, max_y]
        min_x, min_y = rand_points[:,0]-c_w//2, rand_points[:,1]-c_h//2
        max_x, max_y = min_x + c_w, min_y + c_h
        assert np.all((min_x>=0) & (min_y>=0) & (max_x<=im_w) & (max_y<=im_h))

        #取homo_img上h_box的四个顶点
        h_p00, h_p01, h_p10, h_p11 = np.stack([h_min_x, h_min_y],axis=1), np.stack([h_max_x, h_min_y],axis=1),\
                                     np.stack([h_min_x, h_max_y],axis=1), np.stack([h_max_x, h_max_y],axis=1)
        #取img上box的四个顶点
        p00, p01, p10, p11 = np.stack([min_x, min_y],axis=1), np.stack([max_x, min_y],axis=1), \
                             np.stack([min_x, max_y],axis=1), np.stack([max_x, max_y],axis=1)

        #将homo_img上h_box四个点映射至img上
        h_p00_inv = cv2.perspectiveTransform(h_p00.astype(np.float32).reshape(-1, 1, 2), H_inv).squeeze().astype(np.int32)
        h_p01_inv = cv2.perspectiveTransform(h_p01.astype(np.float32).reshape(-1, 1, 2), H_inv).squeeze().astype(np.int32)
        h_p10_inv = cv2.perspectiveTransform(h_p10.astype(np.float32).reshape(-1, 1, 2), H_inv).squeeze().astype(np.int32)
        h_p11_inv = cv2.perspectiveTransform(h_p11.astype(np.float32).reshape(-1, 1, 2), H_inv).squeeze().astype(np.int32)

        #通过对h_p00_inv,...,h_p11_inv进行过滤, 移除homo_img中超出边界的(或处在黑色填充区域的)框
        r0 = (h_p00_inv[:,0]>=0) & (h_p00_inv[:,0]<im_w) & (h_p00_inv[:,1]>=0) & (h_p00_inv[:,1]<im_h)
        r1 = (h_p01_inv[:,0]>=0) & (h_p01_inv[:,0]<im_w) & (h_p01_inv[:,1]>=0) & (h_p01_inv[:,1]<im_h)
        r2 = (h_p10_inv[:,0]>=0) & (h_p10_inv[:,0]<im_w) & (h_p10_inv[:,1]>=0) & (h_p10_inv[:,1]<im_h)
        r3 = (h_p11_inv[:,0]>=0) & (h_p11_inv[:,0]<im_w) & (h_p11_inv[:,1]>=0) & (h_p11_inv[:,1]<im_h)
        reserve = r0 & r1 & r2 & r3

        h_p00, h_p01, h_p10, h_p11 = h_p00[reserve], h_p01[reserve], h_p10[reserve], h_p11[reserve]#移除homo_img上box
        p00, p01, p10, p11 = p00[reserve], p01[reserve], p10[reserve], p11[reserve]#移除img上的box
        h_p00_inv, h_p01_inv, h_p10_inv, h_p11_inv = h_p00_inv[reserve], h_p01_inv[reserve],\
                                                     h_p10_inv[reserve], h_p11_inv[reserve]#移除homo_img上映射到img上的box

        img_polygons = [Polygon(points) for points in zip(p00, p01, p11, p10)]
        homo_img_inv_polygons = [Polygon(points) for points in zip(h_p00_inv, h_p01_inv, h_p11_inv, h_p10_inv)]
        pn = np.array([0]*len(img_polygons))#记录正样本对
        for i, (p1, p2), in enumerate(zip(img_polygons, homo_img_inv_polygons)):
            ratio = polygon_inter_union_ratio(p1,p2)
            if ratio>0.75:
                pn[i] = 1
        base_box, positive_box, negative_box, p_idx, n_idx = None, None, None, None, None
        if np.sum(pn)>=1:
            p_idx = np.random.choice(np.where(pn)[0],)
            base_box = [p00[p_idx][0], p00[p_idx][1], p11[p_idx][0], p11[p_idx][1]]
            positive_box=[h_p00[p_idx][0], h_p00[p_idx][1], h_p11[p_idx][0], h_p11[p_idx][1]]
        else:
            return None, None, None

        pn = np.array([0]*len(homo_img_inv_polygons))#记录负样本
        for i, p in enumerate(homo_img_inv_polygons):
            ratio = polygon_inter_union_ratio(img_polygons[p_idx], p)
            if ratio<0.2:
                pn[i] = 1
        if np.sum(pn)>=1:
            n_idx = np.random.choice(np.where(pn)[0],)
            negative_box = [p00[n_idx][0], p00[n_idx][1], p11[n_idx][0], p11[n_idx][1]]
        else:
            return None, None, None

        base_img = img[base_box[1]:base_box[3],base_box[0]:base_box[2],:]
        positive_img = homo_img[positive_box[1]:positive_box[3], positive_box[0]:positive_box[2], :]
        negative_img = img[negative_box[1]:negative_box[3], negative_box[0]:negative_box[2], :]

        if self.is_train:
            positive_img = self.aug.random_aug(positive_img)

        #normalize
        base_img = self.aug.normalize_image(base_img)
        positive_img = self.aug.normalize_image(positive_img)
        negative_img = self.aug.normalize_image(negative_img)

        return base_img, positive_img, negative_img

    def data_collate(self, batch_list):
        for data in batch_list:
            for d in data:
                if d is None:
                    return None, None, None
        #to chw
        batch_list = [[b.transpose(2,0,1) for b in batch] for batch in batch_list]#to CHW

        batch = list(zip(*batch_list))
        batch = [np.stack(b) for b in batch]
        #
        batch = [torch.tensor(b,dtype=torch.float32) for b in batch]

        return batch


if __name__=="__main__":
    sdata = SimDataSet(r".\datasets\test.txt", output_shape=(160,120))
    sdataloader = DataLoader(sdata, batch_size=2, shuffle=True, collate_fn=sdata.data_collate)
    for i, (base, pos, neg) in enumerate(sdataloader):
        if base is None:
            continue
        print(i)
        # cv2.imshow("base", base[1].transpose(1,2,0))
        # cv2.imshow("pos", pos[1].transpose(1,2,0))
        # cv2.imshow("neg", neg[1].transpose(1,2,0))
        #
        # cv2.waitKey()







