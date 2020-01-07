from Parameters import *
from ImageOperations import *

import numpy as np
import pdb
import timeit

def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.grayscale:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    else:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                if params.grayscale:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                else:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                if params.grayscale:
                    img_block = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W].copy()
                else:
                    img_block = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :].copy()

                img_block_color = average_color(img_block)
                index = find_nearest_color(img_block_color, params.small_images_colors)

                if params.grayscale:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W] = params.small_images[index]
                else:
                    img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]

                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces), end='\r')

    elif params.criterion == 'distantaCuloareMedie2':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                if params.grayscale:
                    img_block = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W].copy()
                else:
                    img_block = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :].copy()

                img_block_color = average_color(img_block)

                neighbours = []
                i2 = i - 1
                j2 = j - 1
                if params.grayscale:
                    up = img_mosaic[i2 * H: (i2 + 1) * H, j * W: (j + 1) * W]
                    left = img_mosaic[i * H: (i + 1) * H, j2 * W: (j2 + 1) * W]
                else:
                    up = img_mosaic[i2 * H: (i2 + 1) * H, j * W: (j + 1) * W, :]
                    left = img_mosaic[i * H: (i + 1) * H, j2 * W: (j2 + 1) * W, :]
                for index, img in enumerate(params.small_images):
                    if np.array_equal(img, up) or np.array_equal(img, left):
                        neighbours.append(index)

                index = find_nearest_color2(img_block_color, params.small_images_colors, neighbours)

                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces), end='\r')

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.grayscale:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
        img_covered = np.zeros(params.image_resized.shape, np.uint8)
    else:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape
        img_covered = np.zeros(params.image_resized.shape[:-1], np.uint8)
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    """
    # Netestat
    if params.criterion == 'aleator':
        progress = 0
        while(0 in img_covered):
            i = np.random.randint(params.image_resized.shape[0] - H)
            j = np.random.randint(params.image_resized.shape[1] - W)
            while(0 not in img_covered[i : i + H, j : j + W]):
                i = np.random.randint(params.image_resized.shape[0] - H)
                j = np.random.randint(params.image_resized.shape[1] - W)
            index = np.random.randint(low=0, high=N, size=1)
            img_mosaic[i : i + H, j : j + W, :] = params.small_images[index]
            if params.grayscale:
                img_covered[i : i + H, j : j + W] = np.full(params.small_images[index].shape, 1)
            else:
                img_covered[i : i + H, j : j + W] = np.full(params.small_images[index].shape[:-1], 1)
            print('Building mosaic %.2f%%' % (100 * np.sum(img_covered) / img_covered.size), end='\r')

            # Safe mode = slow progress
            sm_precision = 5;
            if round(100 * np.sum(img_covered) / img_covered.size) == 99:
                debug("\nEntered safe mode\n")
                zero = np.where(img_covered == 0)
                zero = np.transpose(zero)
                while(len(zero) > 0):
                    i, j = zero[np.random.randint(len(zero))]
                    index = np.random.randint(low=0, high=N, size=1)

                    img_dim = img_mosaic[i : i + H, j : j + W, :].shape
                    img_mosaic[i : i + H, j : j + W, :] = params.small_images[index][0][:img_dim[0], :img_dim[1], :]
                    img_covered[i : i + H, j : j + W] = np.full(img_dim[:-1], 1)

                    zero_block = np.where(img_covered[i : i + H, j : j + W] != -1)
                    zero_block = np.transpose(zero_block)
                    zero = np.delete(zero, zero_block, 0)

                    print('Building mosaic %.2f%% (%d pixels left)' % ((100 * np.sum(img_covered) / img_covered.size), len(zero)), end='\r')
                break
            else:
                progress = round(100 * np.sum(img_covered) / img_covered.size, sm_precision)
        print()
    """

    if params.criterion == 'distantaCuloareMedie':
        while(0 in img_covered):
            i = np.random.randint(params.image_resized.shape[0])
            j = np.random.randint(params.image_resized.shape[1])
            while(0 not in img_covered[i : i + H, j : j + W]):
                i = np.random.randint(params.image_resized.shape[0])
                j = np.random.randint(params.image_resized.shape[1])

            if params.grayscale:
                img_block = params.image_resized[i : i + H, j : j + W].copy()
            else:
                img_block = params.image_resized[i : i + H, j : j + W, :].copy()
            img_block_color = average_color(img_block)
            index = find_nearest_color(img_block_color, params.small_images_colors)

            # Out of bounds fix
            if params.grayscale:
                img_dim = img_mosaic[i : i + H, j : j + W].shape
                img_mosaic[i : i + H, j : j + W] = params.small_images[index][:img_dim[0], :img_dim[1]]
                img_covered[i : i + H, j : j + W] = np.full(img_dim, 1)
            else:
                img_dim = img_mosaic[i : i + H, j : j + W, :].shape
                img_mosaic[i : i + H, j : j + W, :] = params.small_images[index][:img_dim[0], :img_dim[1], :]
                img_covered[i : i + H, j : j + W] = np.full(img_dim[:-1], 1)

            print('\rBuilding mosaic %.2f%%' % (100 * np.sum(img_covered) / img_covered.size), end='')

            # Safe mode
            if round(100 * np.sum(img_covered) / img_covered.size) == 99:
                zero = np.where(img_covered == 0)
                zero = np.transpose(zero)
                while(len(zero) > 0):
                    i, j = zero[np.random.randint(len(zero))]

                    if params.grayscale:
                        img_block = params.image_resized[i : i + H, j : j + W].copy()
                    else:
                        img_block = params.image_resized[i : i + H, j : j + W, :].copy()
                    img_block_color = average_color(img_block)
                    index = find_nearest_color(img_block_color, params.small_images_colors)


                    if params.grayscale:
                        img_dim = img_mosaic[i : i + H, j : j + W].shape
                        img_mosaic[i : i + H, j : j + W] = params.small_images[index][:img_dim[0], :img_dim[1]]
                        img_covered[i : i + H, j : j + W] = np.full(img_dim, 1)
                    else:
                        img_dim = img_mosaic[i : i + H, j : j + W, :].shape
                        img_mosaic[i : i + H, j : j + W, :] = params.small_images[index][:img_dim[0], :img_dim[1], :]
                        img_covered[i : i + H, j : j + W] = np.full(img_dim[:-1], 1)

                    zero_block = np.where(img_covered[i : i + H, j : j + W] != -1)
                    zero_block = np.transpose(zero_block)
                    zero = np.delete(zero, zero_block, axis=0)

                    print('Building mosaic %.2f%% (%d pixels left)' % ((100 * np.sum(img_covered) / img_covered.size), len(zero)), end='\r')
                break
        print()

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    if params.grayscale:
        N, H, W = params.small_images.shape
        h, w = params.image_resized.shape
    else:
        N, H, W, C = params.small_images.shape
        h, w, c = params.image_resized.shape

    # params.num_pieces_vertical = params.num_pieces_vertical * 2 + 1
    params.num_pieces_horizontal = params.num_pieces_horizontal + 1
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal
    offset = H // 2 # 14

    """
    # Netestat
    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))
     """
    if params.criterion == 'distantaCuloareMedie':
        num_pieces = (h-H // H - offset) * ((w-W-1) // (W - offset))
        for i in range(offset, h-H, H):
            for j in range((w-W-1) // (W - offset)):
                i2 = i - (j % 2 * offset)
                j2 = j * (W - offset + 1)
                if params.grayscale:
                    img_block = params.image_resized[i2: i2 + H, j2: j2 + W].copy()
                else:
                    img_block = params.image_resized[i2: i2 + H, j2: j2 + W, :].copy()
                img_block_color = average_color_hex(img_block, params.hex_mask, params.grayscale)
                index = find_nearest_color(img_block_color, params.small_images_colors_hex)

                img_replacement = params.small_images[index]
                img_masked = cv.bitwise_and(img_replacement, img_replacement, mask = params.hex_mask)

                dim = img_mosaic[i2: i2 + H, j2: j2 + W].shape
                if params.grayscale:
                    img_mosaic[i2: i2 + H, j2: j2 + W][:dim[0], :dim[1]] = img_masked[:dim[0], :dim[1]] + \
                        cv.bitwise_and(
                            img_mosaic[i2: i2 + H, j2: j2 + W],
                            img_mosaic[i2: i2 + H, j2: j2 + W],
                            mask=(1 - params.hex_mask[:dim[0], :dim[1]]))
                else:
                    img_mosaic[i2: i2 + H, j2: j2 + W, :][:dim[0], :dim[1]] = img_masked[:dim[0], :dim[1]] + \
                        cv.bitwise_and(
                            img_mosaic[i2: i2 + H, j2: j2 + W, :],
                            img_mosaic[i2: i2 + H, j2: j2 + W, :],
                            mask=(1 - params.hex_mask[:dim[0], :dim[1]]))
                print('Building mosaic %.2f%%' % (100 * (i * ((w-W-1) // (W - offset)) + j) / num_pieces), end='\r')

    elif params.criterion == 'distantaCuloareMedie2':
        num_pieces = (h-H // H - offset) * ((w-W-1) // (W - offset))
        for i in range(offset, h-H, H):
            for j in range((w-W-1) // (W - offset + 1)):
                i2 = i - (j % 2 * offset)
                j2 = j * (W - offset + 1)
                if params.grayscale:
                    img_block = params.image_resized[i2: i2 + H, j2: j2 + W].copy()
                else:
                    img_block = params.image_resized[i2: i2 + H, j2: j2 + W, :].copy()
                img_block_color = average_color_hex(img_block, params.hex_mask, params.grayscale)

                neighbours = []
                i_up = i2 - H
                i_up_side = i2 - offset
                i_down_side = i2 + offset
                j_left = j2 - (W - offset + 1)
                j_right = j2 + (W - offset + 1)

                if params.grayscale:
                    up = img_mosaic[i_up: i_up + H, j2: j2 + W]
                    up_left = img_mosaic[i_up_side: i_up_side + H, j_left: j_left + W]
                    up_right = img_mosaic[i_up_side: i_up_side + H, j_right: j_right + W]
                    down_left = img_mosaic[i_down_side: i_down_side + H, j_left: j_left + W]
                else:
                    up = img_mosaic[i_up: i_up + H, j2: j2 + W, :]
                    up_left = img_mosaic[i_up_side: i_up_side + H, j_left: j_left + W, :]
                    up_right = img_mosaic[i_up_side: i_up_side + H, j_right: j_right + W, :]
                    down_left = img_mosaic[i_down_side: i_down_side + H, j_left: j_left + W, :]

                neighbours_img = []
                if(up.shape == params.small_images[0].shape):
                    neighbours_img.append(up)
                if(up_left.shape == params.small_images[0].shape):
                    neighbours_img.append(up_left)
                if(up_right.shape == params.small_images[0].shape):
                    neighbours_img.append(up_right)
                if(down_left.shape == params.small_images[0].shape):
                    neighbours_img.append(down_left)

                for index, img in enumerate(params.small_images):
                    for neighbour in neighbours_img:
                        img2 = cv.bitwise_and(img, img, mask=params.hex_mask)
                        neighbour2 = cv.bitwise_and(neighbour, neighbour, mask=params.hex_mask)
                        if np.array_equal(img2, neighbour2):
                            neighbours.append(index)

                index = find_nearest_color2_hex(img_block_color, params.small_images_colors_hex, neighbours)

                img_replacement = params.small_images[index]
                img_masked = cv.bitwise_and(img_replacement, img_replacement, mask=params.hex_mask)

                dim = img_mosaic[i2: i2 + H, j2: j2 + W].shape
                if params.grayscale:
                    img_mosaic[i2: i2 + H, j2: j2 + W][:dim[0], :dim[1]] = img_masked[:dim[0], :dim[1]] + \
                        cv.bitwise_and(
                            img_mosaic[i2: i2 + H, j2: j2 + W],
                            img_mosaic[i2: i2 + H, j2: j2 + W],
                            mask=(1 - params.hex_mask[:dim[0], :dim[1]]))
                else:
                    img_mosaic[i2: i2 + H, j2: j2 + W, :][:dim[0], :dim[1]] = img_masked[:dim[0], :dim[1]] + \
                        cv.bitwise_and(
                            img_mosaic[i2: i2 + H, j2: j2 + W, :],
                            img_mosaic[i2: i2 + H, j2: j2 + W, :],
                            mask=(1 - params.hex_mask[:dim[0], :dim[1]]))

                print('Building mosaic %.2f%%' % (100 * (i * ((w-W-1) // (W - offset)) + j) / num_pieces), end='\r')

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic

