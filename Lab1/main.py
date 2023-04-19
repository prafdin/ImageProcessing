import os


from scipy import signal
from enum import Enum
from functools import partial

import matplotlib.pyplot as plt
import numpy
import numpy as np
from PIL import Image
from matplotlib import gridspec
import cv2 as cv

MAX_DEC_VALUE_FOR_BYTE = 255


def read_image(img_path) -> numpy.ndarray:
    return np.array(Image.open(img_path))


def apply_func_to_img(img: np.ndarray, transformation_table) -> np.ndarray:
    return transformation_table[img]


def draw_plots(title, img_before, img_after, transformation_table):
    fig = plt.figure(num=title, figsize=(10, 10))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5, wspace=1)
    gs = gridspec.GridSpec(3, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('Original image')
    ax00.imshow(img_before, cmap='gray', vmin=0, vmax=255)

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_title('Image after applying processing')
    ax01.imshow(img_after, cmap='gray', vmin=0, vmax=255)

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_title('Image histogram before processing')
    ax10.hist(img_before.ravel(), range=(0, MAX_DEC_VALUE_FOR_BYTE))

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_title('Image histogram after processing')
    ax11.hist(img_after.ravel(), range=(0, MAX_DEC_VALUE_FOR_BYTE))

    ax20 = fig.add_subplot(gs[2, :])
    ax20.set_title('Transform function')
    ax20.plot(transformation_table)


class Waveform(Enum):
    @staticmethod
    def triangular(args):
        return signal.sawtooth(args, 0.5)

    HARMONIC = partial(np.sin)
    SQUARE = partial(signal.square)
    TRIANGULAR = partial(triangular.__func__)
    SAWTOOTH = partial(signal.sawtooth)


def main():
    dir_path = "./assets"
    img_name = "04_boat.tif"
    img_path = os.path.join(dir_path, img_name)
    img = read_image(img_path)

    threshold = 130
    binary_transformation_table = np.concatenate(
        (np.zeros(threshold), np.ones(MAX_DEC_VALUE_FOR_BYTE - threshold) * MAX_DEC_VALUE_FOR_BYTE)
    )

    binary_img = apply_func_to_img(img, binary_transformation_table)
    draw_plots("Image binarization", img, binary_img, binary_transformation_table)

    min_value = min(img.ravel())
    max_value = max(img.ravel())
    a = MAX_DEC_VALUE_FOR_BYTE / (max_value - min_value)
    b = -(MAX_DEC_VALUE_FOR_BYTE * min_value) / (max_value - min_value)
    contrasting_transformation_table = np.array([a * i + b for i in range(MAX_DEC_VALUE_FOR_BYTE)])
    contrasting_img = apply_func_to_img(img, contrasting_transformation_table)
    draw_plots("Image contrasting", img, contrasting_img, contrasting_transformation_table)

    img_hist, bins = np.histogram(img.ravel(), bins=MAX_DEC_VALUE_FOR_BYTE + 1, range=[0, MAX_DEC_VALUE_FOR_BYTE])
    cdf = np.cumsum(img_hist)
    cdf = (cdf - min(cdf)) / (max(cdf) - min(cdf))
    g_min, g_max = 0, MAX_DEC_VALUE_FOR_BYTE
    equalization_transformation_table = np.array(
        [(g_max - g_min) * cdf[i] + g_min for i in range(MAX_DEC_VALUE_FOR_BYTE)])
    equalized_img_custom = apply_func_to_img(img, equalization_transformation_table)
    draw_plots("Image equalization (custom)", img, equalized_img_custom, equalization_transformation_table)
    print(f"Image histogram: {img_hist}")
    assert bins[0] == 0 and bins[-1] == 255

    equ = cv.equalizeHist(img)
    equalized_img_lib = equ.astype(np.uint8).reshape(512, 512)
    draw_plots("Image equalization (lib)", img, equalized_img_lib, [])

    peak_count = 4
    sawtooth_transformation_table = Waveform.SAWTOOTH.value(
        2 * np.pi * np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) / (MAX_DEC_VALUE_FOR_BYTE / peak_count)
    )
    sawtooth_transformation_table = (sawtooth_transformation_table + 1) * 255 / 2
    sawtoothed_img = apply_func_to_img(img, sawtooth_transformation_table)
    draw_plots("Image apply sawtooth filter (u)", img, sawtoothed_img, sawtooth_transformation_table)

    square_transformation_table = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (MAX_DEC_VALUE_FOR_BYTE + 1) / 4)/ # here 4 for center impulse
                    MAX_DEC_VALUE_FOR_BYTE
    )
    square_transformation_table = (square_transformation_table + 1) * 255 / 2
    square_img = apply_func_to_img(img, square_transformation_table)
    draw_plots("Image apply square filter (a)", img, square_img, square_transformation_table)

    sawtooth_signal = Waveform.SAWTOOTH.value(
        2 * np.pi * np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) / (
                    MAX_DEC_VALUE_FOR_BYTE)
    )
    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                    MAX_DEC_VALUE_FOR_BYTE + 1) / 4) /
        MAX_DEC_VALUE_FOR_BYTE
    )
    square_with_sawtooth_transformation_table = sawtooth_signal
    square_with_sawtooth_transformation_table[square_signal > 0] = 1
    square_with_sawtooth_transformation_table = (square_with_sawtooth_transformation_table + 1) * 255 / 2
    square_with_sawtooth_img = apply_func_to_img(img, square_with_sawtooth_transformation_table)
    draw_plots("Image apply square with sawtooth filter (b)", img, square_with_sawtooth_img, square_with_sawtooth_transformation_table)

    sawtooth_signal = Waveform.SAWTOOTH.value(
        2 * np.pi * np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) / (
            MAX_DEC_VALUE_FOR_BYTE)
    )
    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) / 2) /
        MAX_DEC_VALUE_FOR_BYTE
    )
    square_with_sawtooth_transformation_table_2 = sawtooth_signal
    square_with_sawtooth_transformation_table_2[square_signal > 0] = 1
    square_with_sawtooth_transformation_table_2 = (square_with_sawtooth_transformation_table_2 + 1) * 255 / 2
    square_with_sawtooth_img = apply_func_to_img(img, square_with_sawtooth_transformation_table_2)
    draw_plots("Image apply square with sawtooth filter #2 (v)", img, square_with_sawtooth_img,
               square_with_sawtooth_transformation_table_2)

    sawtooth_signal = Waveform.SAWTOOTH.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) * 3 / 8) / (
            MAX_DEC_VALUE_FOR_BYTE / 4)
    )
    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) * 3 / 8) /
        MAX_DEC_VALUE_FOR_BYTE,
        duty = 0.25
    )
    single_sawtooth_transformation_table = sawtooth_signal
    single_sawtooth_transformation_table[square_signal < 0] = -1
    single_sawtooth_transformation_table = (single_sawtooth_transformation_table + 1) * 255 / 2
    square_with_sawtooth_img = apply_func_to_img(img, single_sawtooth_transformation_table)
    draw_plots("Image apply square with single sawtooth  (e)", img, square_with_sawtooth_img,
               single_sawtooth_transformation_table)

    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) * 3 / 8) /
        MAX_DEC_VALUE_FOR_BYTE,
        duty=0.25
    )
    sawtooth_signal = Waveform.SAWTOOTH.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) * 3 / 8) / (
                MAX_DEC_VALUE_FOR_BYTE / 4)
    )
    sawtooth_signal[square_signal < 0] = -1
    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE - MAX_DEC_VALUE_FOR_BYTE * 0.375
        )) /
        MAX_DEC_VALUE_FOR_BYTE,
        duty=0.375
    )
    single_sawtooth_with_impulse_transformation_table = sawtooth_signal + square_signal

    single_sawtooth_with_impulse_transformation_table = (single_sawtooth_with_impulse_transformation_table + 1) * 255 / 2
    single_sawtooth_with_impulse_img = apply_func_to_img(img, single_sawtooth_with_impulse_transformation_table)
    draw_plots("Image apply square with single sawtooth and impulse (g)", img, single_sawtooth_with_impulse_img,
               single_sawtooth_with_impulse_transformation_table)

    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) * 3 / 8) /
        MAX_DEC_VALUE_FOR_BYTE,
        duty=0.25
    )
    sawtooth_signal = Waveform.SAWTOOTH.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE + 1) * 3 / 8) / (
                MAX_DEC_VALUE_FOR_BYTE / 4)
    )
    sawtooth_signal[square_signal < 0] = -1
    square_signal = Waveform.SQUARE.value(
        2 * np.pi * (np.linspace(0, MAX_DEC_VALUE_FOR_BYTE, MAX_DEC_VALUE_FOR_BYTE + 1) - (
                MAX_DEC_VALUE_FOR_BYTE - MAX_DEC_VALUE_FOR_BYTE * 0.375
        )) /
        MAX_DEC_VALUE_FOR_BYTE,
        duty=0.375
    )
    single_sawtooth_with_impulse_transformation_table = sawtooth_signal + square_signal
    single_sawtooth_with_impulse_transformation_table = np.array(list(reversed(single_sawtooth_with_impulse_transformation_table)))

    single_sawtooth_with_impulse_transformation_table = (single_sawtooth_with_impulse_transformation_table + 1) * 255 / 2
    single_sawtooth_with_impulse_img = apply_func_to_img(img, single_sawtooth_with_impulse_transformation_table)
    draw_plots("Image apply square with single sawtooth and impulse (d)", img, single_sawtooth_with_impulse_img,
               single_sawtooth_with_impulse_transformation_table)

    plt.show()


if __name__ == '__main__':
    main()
