import scipy.ndimage
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import show
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from skimage.io import imshow


def chess_field(shape, size):
    return (np.indices(shape) // size).sum(axis=0) % 2

def white_noise(d, image, mask_linear, mask_median, w, h):
    dispersion_image = np.var(image)
    noise_image = random_noise(image, var=dispersion_image / d)
    noise = noise_image - image
    print(f"D for white noise in signal/noise ration {d} = ", np.var(noise * 255))

    plt.figure(figsize=(15, 15))
    plt.subplot(231)
    plt.title("Original image")
    plt.imshow(image*255, cmap='gray', vmin=0, vmax=255)

    plt.subplot(232)
    plt.title(f"White noise d = {d}")
    plt.imshow(noise*255 + 128, cmap='gray', vmin=0, vmax=255)

    plt.subplot(233)
    plt.title("Image after applying white noise")
    plt.imshow(noise_image*255, cmap='gray', vmin=0, vmax=255)

    plt.subplot(234)
    plt.title("Image after applying linear filter")
    img_after_applying_linear_filter, mse_linear_filter, noise_reduction_koef_linear_filter = linear_filter(noise_image,
                                                                                                            mask_linear,
                                                                                                            w, h, image)
    print(f"[White noise] MSE for linear filter with signal/noise ration {d}:", mse_linear_filter)
    print(f"[White noise] Reduction koef for linear filter with signal/noise ration {d}:", noise_reduction_koef_linear_filter)
    plt.imshow(img_after_applying_linear_filter*255, cmap='gray', vmin=0, vmax=255)

    plt.subplot(235)
    plt.title("Image after applying median filter")
    img_after_applying_median_filter, mse_median_filter, noise_reduction_koef_median_filter = custom_median_filter(noise_image,
                                                                                                                   mask_median,
                                                                                                                   w, h, image)
    print(f"[White noise] MSE for median filter with signal/noise ration {d}:", mse_median_filter)
    print(f"[White noise] Reduction koef for median filter with signal/noise ration {d}:", noise_reduction_koef_median_filter)
    plt.imshow(img_after_applying_median_filter*255, cmap='gray', vmin=0, vmax=255)



def impulse_noise(p, image, mask_linear, mask_median, w, h):
    noise_image = random_noise(image, mode='s&p', amount=p)
    noise = noise_image - image

    print(f"D for impulse noise with {p} intensity = ", np.var(noise * 255))

    plt.figure(figsize=(15, 15))
    plt.subplot(231)
    plt.title("Original image")
    plt.imshow(image * 255, cmap='gray', vmin=0, vmax=255)

    plt.subplot(232)
    plt.title(f"Impulse noise p = {p}")
    plt.imshow(noise * 255+128, cmap='gray', vmin=0, vmax=255)

    plt.subplot(233)
    plt.title("Image after applying impulse noise")
    plt.imshow(noise_image * 255, cmap='gray', vmin=0, vmax=255)

    plt.subplot(234)
    plt.title("Image after applying linear filter")
    img_after_applying_linear_filter, mse_linear_filter, noise_reduction_koef_linear_filter = linear_filter(noise_image,
                                                                                                            mask_linear,
                                                                                                            w, h, image)
    print(f"[Impulse noise] MSE for linear filter with {p} intensity:", mse_linear_filter)
    print(f"[Impulse noise] Reduction koef for linear filter with {p} intensity:",
          noise_reduction_koef_linear_filter)
    plt.imshow(img_after_applying_linear_filter * 255, cmap='gray', vmin=0, vmax=255)

    plt.subplot(235)
    plt.title("Image after applying median filter")
    img_after_applying_median_filter, mse_median_filter, noise_reduction_koef_median_filter = custom_median_filter(
        noise_image,
        mask_median,
        w, h, image)
    print(f"[Impulse noise] MSE for median filter with {p} intensity:", mse_median_filter)
    print(f"[Impulse noise] Reduction koef for median filter with {p} intensity:",
          noise_reduction_koef_median_filter)
    plt.imshow(img_after_applying_median_filter* 255, cmap='gray', vmin=0, vmax=255)


def linear_filter(noise_image, mask, w, h, orig_image):
    processed_image = convolve2d(noise_image, mask, boundary='symm', mode='same')
    mse = ((np.sum(processed_image *255) - np.sum(orig_image * 255)) ** 2) / w / h
    noise_reduction_koef = np.mean((processed_image  * 255 - orig_image * 255) ** 2) / np.mean((noise_image * 255 - orig_image * 255) ** 2)
    return processed_image, mse, noise_reduction_koef

def custom_median_filter(noise_image, mask, w, h, orig_image):
    processed_image = scipy.ndimage.median_filter(noise_image, footprint=mask)
    error_dispersion = ((np.sum(processed_image*255) - np.sum(orig_image * 255)) ** 2) / w / h
    noise_reducation_koef = np.mean((processed_image * 255 - orig_image * 255) ** 2) / np.mean((noise_image * 255 - orig_image * 255) ** 2)
    return processed_image, error_dispersion, noise_reducation_koef

if __name__ == '__main__':
    w = 128
    h = 128
    cell_size = 16
    board = np.ndarray((w, h))
    pattern = chess_field((w,h), cell_size)
    board[pattern == 0] = 96/255
    board[pattern == 1] = 160/255
    print("[Original image] D:", np.var(board*255))

    mask_linear = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    mask_median = np.array([[0, 1, 0], [1, 3, 1], [0, 1, 0]])
    white_noise(1, board, mask_linear, mask_median, w, h)
    white_noise(10, board, mask_linear, mask_median, w, h)

    impulse_noise(0.1, board, mask_linear, mask_median, w, h)
    impulse_noise(0.3, board, mask_linear, mask_median, w, h)

    plt.show()