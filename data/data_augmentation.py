import os
import cv2
import argparse
import threading
import numpy as np
import numpy.random as random

from glob import glob

""" Implemented by Jarvis73 - Jiawei Zhang """
""" Github link: https://github.com/Jarvis73/Moving-Least-Squares """

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='dataroot path')
    parser.add_argument('--save_path', required=True, help='save path')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--warp', action='store_true')
    parser.add_argument('--spray', action='store_true')
    parser.add_argument('--nsize', type=tuple, default=512)
    return parser.parse_args()


def mls_rigid_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Exchange p and q and hence we transform destination pixels to the corresponding source pixels.
    p, q = q, p

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width * density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height * density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)  # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))  # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1) ** alpha  # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)  # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w  # [2, grow, gcol]
    phat = reshaped_p - pstar  # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)  # [ctrls, 1, 1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0], ...]  # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1, ...] = -neg_phat_verti[:, 1, ...]
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)  # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)  # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar  # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0], ...]  # [2, grow, gcol]
    neg_vpstar_verti[1, ...] = -neg_vpstar_verti[1, ...]
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)  # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)  # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)  # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2),
                  reshaped_mul_right.transpose(0, 3, 4, 1, 2))  # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))  # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)  # [2, grow, gcol]
    qhat = reshaped_q - qstar  # [2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)  # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)  # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)  # [2, grow, gcol]
    norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)  # [1, grow, gcol]
    norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)  # [1, grow, gcol]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar + qstar  # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16),
                                       (np.arange(grow) / density).astype(np.int16))
    # transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    transformed_image[new_gridX, new_gridY] = image[tuple(transformers.astype(np.int16))]

    return transformed_image


def get_random_points(h, w, num=4, scope=50):
    p = random.randint(0, min(h, w), [num, 2])
    q = np.zeros([num, 2])
    for i in range(num):
        dd = [0, 0]
        dd[0] = random.randint(max(-scope, -p[i, 0]), min(scope, h - p[i, 0] - 1))
        dd[1] = random.randint(max(-scope, -p[i, 1]), min(scope, w - p[i, 1] - 1))
        q[i] = p[i] + dd
    return p, q


def warp(file, dataroot, save_path):
    img = cv2.imread(file)
    h, w = img.shape[:2]
    p, q = get_random_points(h, w)
    warped_img = mls_rigid_deformation(img, p, q)
    cv2.imwrite(file.replace(dataroot, save_path), warped_img)


def resize_and_padding(file, dataroot, save_path, nsize=512):
    img = cv2.imread(file)

    height, width, c = img.shape
    if (width > height):
        img = cv2.resize(img, (nsize, int(nsize / width * height)), interpolation=cv2.INTER_AREA)
        new_width = nsize
        new_height = int(nsize / width * height)
    else:
        img = cv2.resize(img, (int(nsize / height * width), nsize), interpolation=cv2.INTER_AREA)
        new_width = int(nsize / height * width)
        new_height = nsize
    padded = np.full((nsize, nsize, 3), 255, dtype=np.uint8)
    padded[(nsize-new_height)//2:(nsize+new_height)//2, (nsize-new_width)//2:(nsize+new_width)//2, :] = img[0:int(new_height), 0:int(new_width), :]

    cv2.imwrite(file.replace(dataroot, save_path), padded)


def drawLine(img, width=80):
    while True:
        x1, y1, x2, y2 = random.randint(0, 511, 4)
        if x1 == x2 and y1 == y2:
            continue
        mx, my = (x1+x2) // 2, (y1+y2) // 2
        if img[mx, my].sum() < 730:
            break
    bgr = (img[mx, my][0].item(), img[mx, my][1].item(), img[mx, my][2].item())
    len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dir_x = (x2 - x1) / len
    dir_y = (y2 - y1) / len

    radius = width // 2
    stroke_color = np.full((512, 512, 3), bgr, dtype=np.uint8)
    beta = np.zeros((512, 512))
    for i in range(int(len)):
         center = (round(x1+dir_x), round(y1+dir_y))
         x1 += dir_x
         y1 += dir_y
         cv2.circle(beta, center, radius=radius, color=1, thickness=radius)
    center = width // 2
    kernel = np.zeros((width, width))
    for x in range(width):
        for y in range(width):
            dx, dy = x - center, y - center
            kernel[y, x] = np.exp(-(dx*dx+dy*dy) / (500*center))
    kernel /= kernel.sum()
    beta = cv2.filter2D(beta, -1, kernel) * 0.975
    beta = np.expand_dims(beta, 2).repeat(3, axis=2)
    alpha = 1. - beta

    dst = (alpha * img + beta * stroke_color).astype(np.uint8)
    return dst


def spray(file, dataroot, save_path, stroke_num=2):
    img = cv2.imread(file)
    for i in range(stroke_num):
        img = drawLine(img)
    cv2.imwrite(file.replace(dataroot, save_path), img)


def processing(thread_id, opt, img_files):
    data_size = len(img_files)
    dataroot = opt.dataroot
    save_path = opt.save_path

    if opt.resize:
        for i in range(data_size):
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))
            resize_and_padding(img_files[i], dataroot, save_path)
    elif opt.warp:
        for i in range(data_size):
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))
            warp(img_files[i], dataroot, save_path)
    elif opt.spray:
        for i in range(data_size):
            if i % 5000 == 0:
                print('id:{}, step: [{}/{}]'.format(thread_id, i, data_size))
            spray(img_files[i], dataroot, save_path)
    else:
        raise ModuleNotFoundError('Please select a pre-processing util.')


def create_threads(opt):
    dataroot = opt.dataroot
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    img_files = glob(os.path.join(dataroot, '*.jpg'))
    data_size = len(img_files)
    print('total data size: {}'.format(data_size))
    thread_num = 8
    thread_size = data_size // thread_num

    threads = []
    for t in range(thread_num):
        if t == thread_num - 1:
            thread = threading.Thread(target=processing, args=(t, opt, img_files[t*thread_size: ]))
        else:
            thread = threading.Thread(target=processing, args=(t, opt, img_files[t*thread_size: (t+1)*thread_size]))
        threads.append(thread)
    for t in threads:
        t.start()
    thread.join()


if __name__ == '__main__':
    opt = get_options()
    create_threads(opt)
