#-*-coding:utf8-*-
import cv2
import numpy as np
from math import pi
from scipy import stats


def correct_homo_mat(H, w, h):
    """
    对透视矩阵进行修正，对输出图像的宽度高度进行预测
    使得输出图时完整的
    :param H (_type_): homography matrix between two images, (ndarray, (3, 3))
    :param w (_type_): image width
    :param h (_type_): image height
    :return
        H (_type_): correct homography matrix, (ndarray, (3, 3))
        correct_w: projection width
        correct_h: projection height
    """
    corner_pts = np.array([[[0, 0], [w, 0], [0, h], [w, h]]], dtype=np.float32)
    min_out_w, min_out_h = cv2.perspectiveTransform(corner_pts, H)[0].min(axis=0).astype(np.int)
    H[0, :] -= H[2, :] * min_out_w
    H[1, :] -= H[2, :] * min_out_h
    correct_w, correct_h = cv2.perspectiveTransform(corner_pts, H)[0].max(axis=0).astype(np.int)

    return H, correct_w, correct_h

def sample_homography(shape):
    '''
    获取一个用于仿射变换的随机变化矩阵
    :param shape:[W, H], 图像大小
    :return:3*3的仿射变换矩阵
    '''

    _config = {'perspective':True, 'scaling':True, 'rotation':True, 'translation':True,
    'n_scales':5, 'n_angles':25, 'scaling_amplitude':0.2, 'perspective_amplitude_x':0.2,
    'perspective_amplitude_y':0.2, 'patch_ratio':0.75, 'max_angle':pi / 2,
    'allow_artifacts': False, 'translation_overflow': 0.}

    # Corners of the input patch
    margin = (1 - _config['patch_ratio']) / 2
    pts1 = margin + np.array([[0, 0],
                              [0, _config['patch_ratio']],
                              [_config['patch_ratio'], _config['patch_ratio']],
                              [_config['patch_ratio'], 0]])
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if _config['perspective']:
        if not _config['allow_artifacts']:
            perspective_amplitude_x = min(_config['perspective_amplitude_x'], margin)
            perspective_amplitude_y = min(_config['perspective_amplitude_y'], margin)
        else:
            perspective_amplitude_x = _config['perspective_amplitude_x']
            perspective_amplitude_y = _config['perspective_amplitude_y']

        tnorm_y = stats.truncnorm(-2, 2, loc=0, scale=perspective_amplitude_y/2)
        tnorm_x = stats.truncnorm(-2, 2, loc=0, scale=perspective_amplitude_x/2)
        perspective_displacement = tnorm_y.rvs(1)
        h_displacement_left = tnorm_x.rvs(1)
        h_displacement_right = tnorm_x.rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if _config['scaling']:
        mu, sigma = 1, _config['scaling_amplitude']/2
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
        tnorm_s = stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)
        scales = tnorm_s.rvs(_config['n_scales'])
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if _config['allow_artifacts']:
            valid = np.arange(_config['n_scales'])  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    # Random translation
    if _config['translation']:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if _config['allow_artifacts']:
            t_min += _config['translation_overflow']
            t_max += _config['translation_overflow']
        pts2 += np.array([np.random.uniform(-t_min[0], t_max[0],1),
                          np.random.uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if _config['rotation']:
        angles = np.linspace(-_config['max_angle'], _config['max_angle'], num=_config['n_angles'])
        angles = np.concatenate((np.array([0.]),angles), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center

        if _config['allow_artifacts']:
            valid = np.arange(_config['n_angles'])  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(low=0, high=valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]

    # Rescale to actual size
    if isinstance(shape, (list, tuple)):
        shape = np.array(shape, dtype=np.float32)#w,h
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    pts1 = pts1.astype("float32")
    pts2 = pts2.astype("float32")

    H_Mat = cv2.getPerspectiveTransform(pts1, pts2)
    #homography = np.linalg.inv(homography)

    return H_Mat#[3,3]



if __name__=='__main__':

    img = cv2.imread(r"..\\a.jpg")
    h, w = img.shape[0:2]
    h_mat = sample_homography(img.shape[0:2][::-1])#w,h
    h_img = cv2.warpPerspective(img, h_mat, (w, h))
    cv2.imshow("img", img)
    cv2.imshow("himg", h_img)


    cv2.waitKey()
    cv2.destroyAllWindows()


