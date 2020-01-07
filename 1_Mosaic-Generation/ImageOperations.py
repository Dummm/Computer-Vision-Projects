import numpy as np

def average_color(img):
    # pixel_count = img.shape[0] * img.shape[1]
    # col_sum = np.sum(img, axis = 0)
    # col_sum = np.sum(col_sum, axis = 0)
    # col_sum = np.divide(col_sum, pixel_count)
    col_sum = np.mean(img, axis=(0, 1))
    return col_sum # BGR

def average_color_hex(img, mask, grayscale = False):
    dim = img.shape
    mask = mask[:dim[0], :dim[1]]

    pixel_count = np.sum(mask)
    if grayscale:
        col_sum = np.multiply(img[:,:], mask)
        col_sum = np.sum(img, axis=(0, 1))
    else:
        col_sum = np.multiply(img[:,:,0], mask)
        col_sum = np.multiply(img[:,:,1], mask)
        col_sum = np.multiply(img[:,:,2], mask)
        col_sum = np.sum(img, axis=(0, 1))

    col_sum = np.divide(col_sum, pixel_count)
    return col_sum # BGR

def color_distance(col1, col2):
    dif = col2 - col1
    pow = np.power(dif, 2)
    sum = np.sum(pow)
    sqr = np.sqrt(sum)
    return sqr

def find_nearest_color(color_orig, color_list):
    color_result = 0
    color_result_dist = color_distance(color_orig, color_list[color_result])

    for i in range(color_list.shape[0]):
        dist = color_distance(color_orig, color_list[i])
        if(dist < color_result_dist):
            color_result = i
            color_result_dist = dist

    return color_result


def find_nearest_color2(color_orig, color_list, neighbours):
    color_result = 0
    while color_result in neighbours:
        color_result = color_result + 1
    color_result_dist = color_distance(color_orig, color_list[color_result])

    for i in range(color_list.shape[0]):
        if i in neighbours:
            continue
        dist = color_distance(color_orig, color_list[i])
        if (dist < color_result_dist):
            color_result = i
            color_result_dist = dist

    return color_result


def find_nearest_color2_hex(color_orig, color_list, neighbours):
    color_result = 0
    while color_result in neighbours:
        color_result = color_result + 1
    color_result_dist = color_distance(color_orig, color_list[color_result])

    for i in range(color_list.shape[0]):
        if i in neighbours:
            continue
        dist = color_distance(color_orig, color_list[i])
        if (dist < color_result_dist):
            color_result = i
            color_result_dist = dist

    return color_result
