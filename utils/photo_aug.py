import cv2
import random
import numpy as np

class PhotoAug:
    '''
    对图像进行无形变增强
    '''
    def __init__(self, ):
        pass

    def blur(self, image):
        img = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0., sigmaY=0.)
        return img

    def gaussian_noise(self, image, mean=0, var=12):
        '''
        高斯噪声
        '''
        img_h, img_w, img_c = image.shape#HWC

        # 根据均值和标准差生成符合高斯分布的噪声
        gauss = np.random.normal(mean, var, (img_h, img_w, img_c))
        # 给图片添加高斯噪声
        noisy_img = image + gauss
        # 设置图片添加高斯噪声之后的像素值的范围
        noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
        noisy_img = np.floor(noisy_img)
        noisy_img = noisy_img.astype(np.uint8)
        return noisy_img

    def sp_noise(self, image, amount=0.01):
        '''
        椒盐噪声
        '''
        output = image.copy()
        threshold = 1 - amount
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdm = random.random()
                if rdm < amount:
                    output[i][j] = 0
                elif rdm > threshold:
                    output[i][j] = 255
        return output

    def flip(self, image, mode="lr"):
        '''镜像翻转
        lr:左右反转
        ud:上下翻转
        '''
        img = None
        if mode=="lr":
            img = cv2.flip(image, 1)#水平翻转
        else:
            img = cv2.flip(image, 0)
        return img

    def normalize_image(self, image,  mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        '''
        image: opencv image with shape HWC
        当mean或std为None时，返回image/255
        '''
        if mean is None:
            return image/255.

        if isinstance(mean,(list, tuple)):
            mean = np.array(mean)
        if isinstance(std,(list, tuple)):
            std = np.array(std)
        mean = mean.reshape(1,1,-1)#1,1,3
        std = std.reshape(1,1,-1)#1,1,3
        image = (image/255. - mean)/std
        return image

    def random_aug(self, image):
        '''
        输入opencv image，输出增强后的图像
        '''
        _image = image.copy()
        if random.uniform(0, 1)>0.5:
            _image = self.blur(_image)
        if random.uniform(0, 1)>0.5:
            _image = self.gaussian_noise(_image)
        if random.uniform(0, 1)>0.5:
            if random.uniform(0,1)>0.5:
                _image = self.flip(_image, "lr")
            else:
                _image = self.flip(_image, "ud")

        return _image


if __name__=="__main__":
    pa = PhotoAug()
    for i in range(100):
        image = cv2.imread(r'..\a.jpg', cv2.IMREAD_COLOR)
        image_aug = pa.random_aug(image)
        cv2.imshow("raw", image)
        cv2.imshow("noise", image_aug)
        cv2.waitKey()

