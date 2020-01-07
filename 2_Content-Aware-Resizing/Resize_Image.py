import sys
import cv2 as cv
import numpy as np

from Parameters import *
from SelectPath import *

import pdb

DEC_VALUE = 10000

def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea imaginii

    img_gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY);

    grad_x = cv.Sobel(
        img_gray_scale,
        ddepth = cv.CV_16S,
        dx = 1, dy = 0,
        borderType = cv.BORDER_CONSTANT
    );
    grad_y = cv.Sobel(
        img_gray_scale,
        ddepth = cv.CV_16S,
        dx = 0, dy = 1,
        borderType = cv.BORDER_CONSTANT
    );

    E = np.array(abs(grad_x) + abs(grad_y), dtype=np.int32);

    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    cv.imshow('path', np.uint8(new_image))
    cv.waitKey(100)


def delete_path(img, pathway):
    """
     elimina drumul vertical din imagine
    :param img: imaginea initiala
    :pathway - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    if img.ndim == 3:
        updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    else:
        updated_img = np.zeros((img.shape[0], img.shape[1] - 1), np.uint8)
    for i in range(img.shape[0]):
        col = pathway[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        # completati aici codul vostru
        updated_img[i, col:] = img[i, col+1:].copy()
    return updated_img

def insert_path(img, pathway):
    updated_img = np.zeros((img.shape[0], img.shape[1] + 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col = pathway[i][1]
        updated_img[i, :col] = img[i, :col].copy()
        updated_img[i, col] = np.mean(img[i, col:col+2], axis=(0))
        updated_img[i, col+1:] = img[i, col:].copy()
    return updated_img


def decrease_width(params: Parameters, num_pixels, object=None):
    img = params.image.copy() # copiaza imaginea originala

    # calculeaza energia dupa ecuatia (1) din articol
    # E = compute_energy(img)

    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels), end='\r')

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        if object != None:
            (x, y, w, h) = object
            E[y: y + h, x : x + w - i] = E[y: y + h, x : x + w - i] - DEC_VALUE
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)
        # E = delete_path(E, path)

    cv.destroyAllWindows()
    return img

def decrease_height(params: Parameters, num_pixels, object=None):
    img = params.image.copy() # copiaza imaginea originala
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE) # Rotirea imaginii

    # calculeaza energia dupa ecuatia (1) din articol
    # E = compute_energy(img)

    for i in range(num_pixels):
        print('Eliminam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixels), end='\r')

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        if object != None:
            (x, y, w, h) = object
            x = (img.shape[0] - 1) - x
            # E[x - w : x, y : y + h - i] = - 1000
            E[x - w : x, y : y + h - i] = E[x - w : x, y : y + h - i] - DEC_VALUE
            # E[x - w : x, y : y + h - i] = np.amin(E)
        path = select_path(E, params.method_select_path)
        # plt.imshow(E)
        # plt.show()
        if params.show_path:
            img2 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            path2 = [(y, (img2.shape[1] - 1) - x) for (x, y) in path]
            # path2.reverse()
            show_path(img2, path2, params.color_path)
        img = delete_path(img, path)
        # E = delete_path(E, path)

    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    cv.destroyAllWindows()
    return img


def fix_paths(paths):
    # Drumurile sunt adaugate de la dreapta la stanga
    # Daca doua drumuri, i si j, se intersecteaza,
    #   al doilea se misca cu un pixel la dreapta de la punctul de intersectie in jos
    for i in range(len(paths)):
        for j in range(i, len(paths)):
            for k in range(len(paths[i])):
                if paths[i][k][1] < paths[j][k][1]:
                    paths[j][k] = (paths[j][k][0], paths[j][k][1] + 1)

def increase_width(params: Parameters, num_pixels, object=None):
    img = params.image.copy()

    # Generarea drumurilor
    paths = []
    for i in range(num_pixels):
        E = compute_energy(img)
        # if object != None:
        #     (x, y, w, h) = object
        #     E[y: y + h, x : x + i + 1] = E[y: y + h, x : x + i + 1] - 1000
        path = select_path(E, params.method_select_path)
        paths.append(path)
        img = delete_path(img, path)
    # Sortarea drumurilor de la dreapta la stanga in functie de primul punct
    paths.sort(key=lambda x: x[0][1], reverse=True)

    fix_paths(paths)

    img = params.image.copy() # copiaza imaginea originala
    for i in range(num_pixels):
        print('Adaugam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels), end='\r')

        # calculeaza energia dupa ecuatia (1) din articol
        # E = compute_energy(img)
        path = paths[i]
        if params.show_path:
            show_path(img, path, params.color_path)
        img = insert_path(img, path)

    cv.destroyAllWindows()
    return img

def increase_height(params: Parameters, num_pixels, object=None):
    img = params.image.copy()
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE) # Rotirea imaginii
    paths = []
    for i in range(num_pixels):
        E = compute_energy(img)
        # if object != None:
        #     (x, y, w, h) = object
        #     x = (img.shape[0] - 1) - x
        #     E[x - w : x, y : y + i + 1] = E[x - w : x, y : y + i + 1] - 1000
        path = select_path(E, params.method_select_path)
        paths.append(path)
        img = delete_path(img, path)
    paths.sort(key=lambda x: (img.shape[0] - x[-1][1]), reverse=True)

    fix_paths(paths)

    img = params.image.copy() # copiaza imaginea originala
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE) # Rotirea imaginii
    for i in range(num_pixels):
        print('Adaugam drumul orizontal numarul %i dintr-un total de %d.' % (i + 1, num_pixels), end='\r')

        # calculeaza energia dupa ecuatia (1) din articol
        # E = compute_energy(img)
        path = paths[i]
        if params.show_path:
            img2 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
            path2 = [(y, (img2.shape[1] - 1) - x) for (x, y) in path]
            # path2.reverse()
            show_path(img2, path2, params.color_path)
        img = insert_path(img, path)

    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    cv.destroyAllWindows()
    return img


def amplify_content(params: Parameters, scale):
    img = params.image.copy()
    img = cv.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
    diff = (img.shape[0] - params.image.shape[0], img.shape[1] - params.image.shape[1])

    img_orig = params.image.copy()
    params.image = img.copy()
    img = decrease_width(params, diff[1])
    params.image = img.copy()
    img = decrease_height(params, diff[0])
    params.image = img_orig.copy()

    return img


def delete_object(params:Parameters):
    img = params.image.copy()
    selection = cv.selectROI(img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    (x, y, w, h) = selection
    img_orig = params.image.copy()

    if w <= h:
        img = decrease_width(params, w, selection)
        params.image = img.copy()
        img = increase_width(params, w, selection)
    else:
        img = decrease_height(params, h, selection)
        params.image = img.copy()
        img = increase_height(params, h, selection)

    params.image = img_orig.copy()

    return img


def resize_image(params: Parameters):
    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image
    elif params.resize_option == 'micsoreazaInaltime':
        # redimensioneaza imaginea pe inaltime
        resized_image = decrease_height(params, params.num_pixels_height)
        return resized_image
    elif params.resize_option == 'maresteLatime':
        # mareste imaginea pe latime
        resized_image = increase_width(params, params.num_pixels_width)
        return resized_image
    elif params.resize_option == 'maresteInaltime':
        # mareste imaginea pe inaltime
        resized_image = increase_height(params, params.num_pixels_height)
        return resized_image
    elif params.resize_option == 'amplificaContinut':
        # amplifica continutul imaginii
        resized_image = amplify_content(params, params.amplification_factor)
        return resized_image
    elif params.resize_option == 'eliminaObiect':
        # elimina obiect din imagine
        resize_image = delete_object(params)
        return resize_image
    else:
        print('The option is not valid!')
        sys.exit(-1)
