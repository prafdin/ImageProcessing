import math
import os

import numpy as np
from PIL import Image, ImageEnhance

from matplotlib import pyplot as plt, gridspec
from scipy.signal import convolve2d
import cv2 as cv

MAX_DEC_VALUE_FOR_BYTE = 255


def show_laplacian_applying(title, orig_img, contours, estimation):
    fig = plt.figure(num=title, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    gs = gridspec.GridSpec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Original image')
    ax00.imshow(orig_img, cmap='gray', vmin=0, vmax=255)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Detected contours')
    ax01.imshow(contours, cmap='gray', vmin=0, vmax=255)

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_title('Laplacian estimation')
    ax10.imshow(estimation, cmap='gray', vmin=0, vmax=255)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_title('Estimation hist')
    ax11.hist(estimation.ravel(), bins='auto')


def show_gradient_applying(title, orig_img, contours, g_x, g_y, estimation):
    fig = plt.figure(num=title, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    gs = gridspec.GridSpec(3, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Original image')
    ax00.imshow(orig_img, cmap='gray', vmin=0, vmax=255)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Detected contours')
    ax01.imshow(contours, cmap='gray', vmin=0, vmax=255)

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_title('x derivative')
    ax10.imshow(g_x + 128, cmap='gray', vmin=0, vmax=255)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_title('y derivative')
    ax11.imshow(g_y + 128, cmap='gray', vmin=0, vmax=255)

    ax20 = fig.add_subplot(gs[2, 0])
    ax20.set_title('Gradient estimation')
    ax20.imshow(estimation, cmap='gray', vmin=0, vmax=255)

    ax21 = fig.add_subplot(gs[2, 1])
    ax21.set_title('Gradient hist')
    ax21.hist(estimation.ravel(), bins='auto')


# https://habr.com/ru/articles/489734/#2d
def rolling_window_2d(a, window_shape, dx=1, dy=1):
    if (len(window_shape) > 2):
        raise Exception("Function supports only 2d window")

    shape = a.shape[:-2] + \
            ((a.shape[-2] - window_shape[0]) // dy + 1,) + \
            ((a.shape[-1] - window_shape[1]) // dx + 1,) + \
            (window_shape[0], window_shape[1])  # sausage-like shape with 2D cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def pad_2d_matrix_to_size(a: np.ndarray, new_xsize, new_ysize, pad_value=0):
    if (len(a.shape) != 2):
        raise Exception("Function works only with 2d matrix")

    current_xsize = a.shape[0]
    current_ysize = a.shape[1]
    delta_x = new_xsize - current_xsize
    delta_y = new_ysize - current_ysize

    if (delta_x < 0 or delta_y < 0):
        raise Exception("The current size of the matrix is greater than the new size")
    d = int(delta_y // 2),
    d = int(delta_y - (delta_y // 2))
    return np.pad(
        a,
        (
            (
                int(delta_x // 2),
                int(delta_x - (delta_x // 2))
            ),
            (
                int(delta_y // 2),
                int(delta_y - (delta_y // 2))
            ),
        ),
        mode='constant',
        constant_values=pad_value
    )


def simple_gradient(img: np.ndarray):
    horizontal_mask = np.array([[-1, 1]])
    vertical_mask = np.array([[-1], [1]])
    g_x = convolve2d(img, horizontal_mask, boundary="symm", mode="same")
    g_y = convolve2d(img, vertical_mask, boundary="symm", mode="same")
    mod_gradient = np.array(np.sqrt((g_x ** 2) + (g_y ** 2))).astype(np.int)

    threshold = 10
    contours = np.ceil((mod_gradient - threshold) / 255) * 255

    show_gradient_applying("Simple gradient", img, contours, g_x, g_y, mod_gradient)


def laplacian(img: np.ndarray):
    m1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    m2 = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]]) / 2
    m3 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) / 3
    l1 = np.array(np.abs(convolve2d(img, m1, boundary="symm", mode="same"))).astype(np.int)
    l2 = np.array(np.abs(convolve2d(img, m2, boundary="symm", mode="same"))).astype(np.int)
    l3 = np.array(np.abs(convolve2d(img, m3, boundary="symm", mode="same"))).astype(np.int)

    threshold = 10
    c1 = np.ceil((l1 - threshold) / 255) * 255
    c2 = np.ceil((l2 - threshold) / 255) * 255
    c3 = np.ceil((l3 - threshold) / 255) * 255

    show_laplacian_applying("Laplacian #1", img, c1, l1)
    show_laplacian_applying("Laplacian #2", img, c2, l2)
    show_laplacian_applying("Laplacian #3", img, c3, l3)


def pruit(img: np.ndarray):
    m1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 6
    m2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 6
    g_x = convolve2d(img, m1, boundary="symm", mode="same")
    g_y = convolve2d(img, m2, boundary="symm", mode="same")
    grad = np.sqrt((g_x ** 2) + (g_y ** 2))

    threshold = 20
    contours = np.ceil((grad - threshold) / 255) * 255

    show_gradient_applying("Pruit method", img, contours, g_x, g_y, grad)


def agreement_laplassian(img: np.ndarray):
    m = np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]) / 3
    l = np.abs(convolve2d(img, m, boundary="symm", mode="same"))

    threshold = 30
    c = np.ceil((l - threshold) / 255) * 255

    show_laplacian_applying("Laplassian via agreement plane", img, c, l)


def rank_filter(title, img: np.ndarray):
    new_x = 3 * (math.ceil(img.shape[0] / 3))
    new_y = 3 * (math.ceil(img.shape[1] / 3))
    img = pad_2d_matrix_to_size(img, new_x, new_y)
    samples = rolling_window_2d(img, [3, 3])

    img_after = np.array([[ np.max(column) - np.min(column) for column in line] for line in samples]).astype(np.int)

    threshold = 20
    borders = np.ceil((img_after - threshold) / 255) * 255

    fig = plt.figure(num=title, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    gs = gridspec.GridSpec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Original image')
    ax00.imshow(img, cmap='gray', vmin=0, vmax=255)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Borders')
    ax01.imshow(np.array(borders), cmap='gray', vmin=0, vmax=255)

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_title('After filter')
    ax10.imshow(np.array(img_after), cmap='gray', vmin=0, vmax=255)

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_title('Hist')
    ax11.hist(np.array(img_after).ravel(), bins='auto')


def apply_canny(title, img: np.ndarray):
    borders = cv.Canny(img, 100, 200)

    fig = plt.figure(num=title, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    gs = gridspec.GridSpec(1, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Original image')
    ax00.imshow(img, cmap='gray', vmin=0, vmax=255)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Borders')
    ax01.imshow(np.array(borders), cmap='gray', vmin=0, vmax=255)



def main():
    dir_path = "./assets"

    img_name = "04_boat.tif"
    img_path = os.path.join(dir_path, img_name)
    image = Image.open(img_path).convert("L")
    image = ImageEnhance.Contrast(image).enhance(2)
    image = np.array(image)

    custom_img = "bykovka_river.png"
    custom_img_path = os.path.join(dir_path, custom_img)
    custom_image = Image.open(custom_img_path).convert("L")
    custom_image = ImageEnhance.Contrast(custom_image).enhance(2)
    custom_image = np.array(custom_image)

    # simple_gradient(image)
    # laplacian(image)
    # pruit(image)
    # agreement_laplassian(image)
    rank_filter("default img", image)
    rank_filter("custom img rank", custom_image)
    apply_canny("custom img canny", custom_image)
    plt.show()


if __name__ == '__main__':
    main()
