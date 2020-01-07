import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb

from AddPiecesMosaic import *
from Parameters import *
from ImageOperations import *


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice H x W x C x N in params
    # pieseMoziac(:,:,:,i) reprezinta piesa numarul i
    images = []
    images_colors = []
    images_colors_hex = []
    files = os.listdir(params.small_images_dir)
    for file in files:
        if params.grayscale:
            img = cv.imread(os.path.join(params.small_images_dir, file), cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(os.path.join(params.small_images_dir, file))
        images.append(img)
        images_colors.append(average_color(img))
        if params.hexagon:
            images_colors_hex.append(average_color_hex(img, params.hex_mask, params.grayscale))
    images = np.array(images)
    images_colors = np.array(images_colors)
    images_colors_hex = np.array(images_colors_hex)
    # citeste imaginile din director

    if params.show_small_images:
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, i * 10 + j + 1)
                # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                im = images[i * 10 + j].copy()
                # BGR to RGB, swap the channels
                im = im[:, :, [2, 1, 0]]
                plt.imshow(im)
        plt.show()

    params.small_images = images
    params.small_images_colors = images_colors
    params.small_images_colors_hex = images_colors_hex


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    image_ratio = params.image.shape[1] / params.image.shape[0]
    if params.grayscale:
        small_images_h, small_images_w = params.small_images[0].shape
    else:
        small_images_h, small_images_w, small_images_c = params.small_images[0].shape
    # debug("image.shape: ", params.image.shape)
    # debug("image_ratio: ", image_ratio)
    # debug("small_images_shape: ", small_images_h, small_images_w)

    mozaic_w = params.num_pieces_horizontal * small_images_w
    mozaic_h = mozaic_w / image_ratio
    # debug("mozaic_w: ", mozaic_w)
    # debug("mozaic_h: ", mozaic_h)

    params.num_pieces_vertical = round(mozaic_h / small_images_h)
    mozaic_h = params.num_pieces_vertical * small_images_h
    # debug("mozaic_h: ", mozaic_h)
    # debug("num_pieces_horizontal: ", params.num_pieces_horizontal)
    # debug("num_pieces_vertical: ", params.num_pieces_vertical)

    # redimensioneaza imaginea
    new_h = mozaic_h
    new_w = mozaic_w
    # debug("new_w: ", new_w)
    # debug("new_h: ", new_h)
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
