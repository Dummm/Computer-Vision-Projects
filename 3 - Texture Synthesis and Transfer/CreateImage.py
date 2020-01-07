from Parameters import *
import numpy as np
import pdb

'''
        Random
'''
def add_random_blocks(params: Parameters, blocks):
    n_blocks_x = int(np.ceil(params.dim_result_image[1] / params.dim_block))
    n_blocks_y = int(np.ceil(params.dim_result_image[0] / params.dim_block))

    if params.image.ndim == 3:
        aux_image = np.zeros((n_blocks_y * params.dim_block, n_blocks_x * params.dim_block, 3), np.uint8)
    else:
        aux_image = np.zeros((n_blocks_y * params.dim_block, n_blocks_x * params.dim_block), np.uint8)

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            cv.imshow('img', aux_image)
            cv.waitKey(100)
            # cv.destroyAllWindows()
            print(" %.2f%%" % ((y * n_blocks_y + x) / (n_blocks_x * n_blocks_y) * 100), end='\r')
            idx = np.random.randint(low=0, high=len(blocks), size=1)[0]
            start_y = y * params.dim_block
            end_y   = start_y + params.dim_block
            start_x = x * params.dim_block
            end_x   = start_x + params.dim_block
            aux_image[start_y: end_y, start_x: end_x] = blocks[idx]

    result_image = aux_image[:params.dim_result_image[0], :params.dim_result_image[1]]
    return result_image


'''
        Overlap
'''
def block_distance(block_a, block_b):
    # block_a = np.int32(block_a)
    # block_b = np.int32(block_b)

    # pixel_count = block_a.size
    # print(block_a.shape, block_b.shape)
    result = cv.absdiff(block_a, block_b)
    result = np.sum(result)
    # result = result / pixel_count

    # cv.imshow('img', block_a)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imshow('img', block_b)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imshow('img', result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # print(result)

    # res = np.subtract(block_a, block_b)
    # res = np.power(res, 2)
    # res = np.power(res, 1/2)
    # res = np.sum(res)
    # res = res / pixel_count

    # return np.uint8(result)
    return result
    # return res

def get_distances(params: Parameters, aux_image, y, x, blocks, mode = None):
    overlap = int(np.round(params.block_overlap * params.dim_block))
    new_dim_block = params.dim_block - overlap

    start_y = y * new_dim_block
    end_y   = start_y + params.dim_block
    start_x = x * new_dim_block
    end_x   = start_x + params.dim_block

    distances = []

    for i in range(len(blocks)):
        # cv.imshow('img',    aux_image[start_y:end_y, start_x:end_x])
        # cv.imshow('img_tl', aux_image[start_y:start_y + overlap, start_x:start_x + overlap])
        # cv.imshow('img_l',  aux_image[start_y + overlap:end_y, start_x:start_x + overlap])
        # cv.imshow('img_t',  aux_image[start_y:start_y + overlap, start_x + overlap:end_x])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        if mode == None:
            # Top left overlap
            distance = block_distance(
                blocks[i][:overlap, :overlap],
                aux_image[start_y:start_y + overlap, start_x:start_x + overlap]
            )
            # Left overlap
            if x != 0:
                # print(overlap)
                distance = distance + block_distance(
                    blocks[i][overlap:, :overlap],
                    aux_image[start_y + overlap:end_y, start_x:start_x + overlap]
                )
            # Top overlap
            if y != 0:
                distance = distance + block_distance(
                    blocks[i][:overlap, overlap:],
                    aux_image[start_y:start_y + overlap, start_x + overlap:end_x]
                )
        else:
            distance = block_distance(blocks[i], aux_image[start_y:end_y, start_x:end_x])

        distances.append(distance)

    return distances

def find_neighbouring_block(params: Parameters, aux_image, y, x, blocks):
    distances = get_distances(params, aux_image, y, x, blocks)
    distances = np.array(distances)

    min_dist = min(distances)
    if min_dist <= 0:
        min_dist = 0.1
    max_dist = min_dist * (1 + params.tolerance)

    candidates = np.where(distances <= max_dist)[0]
    idx = np.random.randint(low=0, high = len(candidates), size=1)[0]

    return candidates[idx]

def add_overlap_blocks(params: Parameters, blocks):
    overlap = int(np.round(params.block_overlap * params.dim_block))
    new_dim_block = params.dim_block - overlap

    n_blocks_x = int(np.floor(params.dim_result_image[1] / new_dim_block))
    n_blocks_y = int(np.floor(params.dim_result_image[0] / new_dim_block))

    if params.image.ndim == 3:
        aux_image = np.zeros((n_blocks_y * new_dim_block + overlap, n_blocks_x * new_dim_block + overlap, 3), np.uint8)
    else:
        aux_image = np.zeros((n_blocks_y * new_dim_block + overlap, n_blocks_x * new_dim_block + overlap), np.uint8)

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            cv.imshow('img', aux_image)
            cv.waitKey(100)
            # cv.destroyAllWindows()
            print(" %.2f%%" % ((y * n_blocks_x + x) / (n_blocks_x * n_blocks_y) * 100), end='\r')

            if y == 0 and x == 0:
                idx = np.random.randint(low=0, high=len(blocks), size=1)[0]
            else:
                idx = find_neighbouring_block(params, aux_image, y, x, blocks)

            start_y = y * new_dim_block
            end_y   = start_y + params.dim_block
            start_x = x * new_dim_block
            end_x   = start_x + params.dim_block

            aux_image[start_y: end_y, start_x: end_x] = blocks[idx]

    result_image = aux_image[:params.dim_result_image[0], :params.dim_result_image[1]]
    return result_image


'''
        Frontier
'''
def find_minimum_cost_path(E):
    D = np.ndarray(E.shape, dtype=E.dtype)
    D[0, :] = E[0, :]
    for i in range(1, D.shape[0]):
        D[i, 0] = E[i, 0] + min(D[i-1, 0], D[i-1, 1])
        for j in range(1, D.shape[1] - 1):
            D[i, j] = E[i, j] + min(D[i-1, j-1], D[i-1, j], D[i-1, j+1])
        D[i, D.shape[1] - 1] = E[i, D.shape[1] - 1] + min(D[i-1, D.shape[1] - 2], D[i-1, D.shape[1] - 1])

    line = D.shape[0] - 1
    column = np.argmin(D[line, :])
    pathway = [(line, column)]

    while pathway[0][0] > 0:
        (line, column) = pathway[0]
        line = line - 1
        if column == 0:
            opt = np.argmin(D[line,column:column+2])
        elif column == E.shape[1]-1:
            opt = np.argmin(D[line,column-1:column+1]) - 1
        else:
            opt = np.argmin(D[line,column-1:column+2]) - 1
        pathway.insert(0, (line, column+opt))

    return pathway

def add_block_based_on_frontier(params: Parameters, aux_image, y, x, block):
    overlap = int(np.round(params.block_overlap * params.dim_block))
    new_dim_block = params.dim_block - overlap

    start_y = y * new_dim_block
    end_y   = start_y + params.dim_block
    start_x = x * new_dim_block
    end_x   = start_x + params.dim_block

    aux_image_left = aux_image[start_y:end_y, start_x:start_x + overlap]
    aux_image_top  = aux_image[start_y:start_y + overlap, start_x:end_x]
    block_left     = block[:, :overlap]
    block_top      = block[:overlap, :]

    E_left = cv.absdiff(aux_image_left, block_left)
    E_left = np.sum(E_left, axis=2)
    E_top  = cv.absdiff(aux_image_top, block_top)
    E_top  = np.sum(E_top, axis=2)

    # M = np.full(block.shape, 255)
    M = np.full(block.shape, 1)
    # M = np.uint8(M)
    M = np.float32(M)
    # cv.imshow('1', np.uint8(255 * M))

    if x != 0:
        path_left = find_minimum_cost_path(E_left)
        for (i, j) in path_left:
            # Faded
            M[i, :j] = 0
            if j >= 0:
                M[i, j] = 0.75
                if j >= 1:
                    M[i, j-1] = 0.50
                    if j >= 2:
                        M[i, j-2] = 0.25
        # cv.imshow('2', np.uint8(255 * M))

    if y != 0:
        E_top = cv.rotate(E_top, cv.ROTATE_90_COUNTERCLOCKWISE)
        path_top  = find_minimum_cost_path(E_top)
        M = cv.rotate(M, cv.ROTATE_90_COUNTERCLOCKWISE)
        for (i, j) in path_top:
            # Faded
            M[i, :j] = 0
            if j >= 0:
                M[i, j] = 0.75
                if j >= 1:
                    M[i, j-1] = 0.50
                    if j >= 2:
                        M[i, j-2] = 0.25
        M = cv.rotate(M, cv.ROTATE_90_CLOCKWISE)
        # cv.imshow('4', np.uint8(255 * M))


    # cv.imshow('3', np.uint8(255 * M))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # cv.imshow('A', np.uint8(aux_image[start_y: end_y, start_x: end_x] * (1 - M)))
    # cv.imshow('B', np.uint8(block * M))
    # cv.imshow('M', np.uint8(255 * M))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    aux_image[start_y: end_y, start_x: end_x] = \
        aux_image[start_y: end_y, start_x: end_x] * (1 - M) + \
        block * M

def add_frontier_blocks(params: Parameters, blocks):
    overlap = int(np.round(params.block_overlap * params.dim_block))
    new_dim_block = params.dim_block - overlap

    n_blocks_x = int(np.floor(params.dim_result_image[1] / new_dim_block))
    n_blocks_y = int(np.floor(params.dim_result_image[0] / new_dim_block))

    if params.image.ndim == 3:
        aux_image = np.zeros((n_blocks_y * new_dim_block + overlap, n_blocks_x * new_dim_block + overlap, 3), np.uint8)
    else:
        aux_image = np.zeros((n_blocks_y * new_dim_block + overlap, n_blocks_x * new_dim_block + overlap), np.uint8)

    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            cv.imshow('img', aux_image)
            cv.waitKey(100)
            # cv.destroyAllWindows()
            print(" %.2f%%" % ((y * n_blocks_x + x) / (n_blocks_x * n_blocks_y) * 100), end='\r')

            if y == 0 and x == 0:
                idx = np.random.randint(low=0, high=len(blocks), size=1)[0]
                start_y = y * new_dim_block
                end_y   = start_y + params.dim_block
                start_x = x * new_dim_block
                end_x   = start_x + params.dim_block
                aux_image[start_y: end_y, start_x: end_x] = blocks[idx]
            else:
                idx = find_neighbouring_block(params, aux_image, y, x, blocks)
                add_block_based_on_frontier(params, aux_image, y, x, blocks[idx])
                # aux_image[start_y: end_y, start_x: end_x] = blocks[idx]

    result_image = aux_image[:params.dim_result_image[0], :params.dim_result_image[1]]
    return result_image


'''
        Transfer
'''
def block_intensity_distance(block_a, block_b):
    a = np.uint8(cv.cvtColor(block_a, cv.COLOR_BGR2GRAY))
    b = np.uint8(cv.cvtColor(block_b, cv.COLOR_BGR2GRAY))
    c = np.uint8(cv.absdiff(a, b))
    s = np.sum(c)
    # cv.imshow('a', a)
    # cv.imshow('b', b)
    # cv.imshow('c', c)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return s

def get_intensities(params:Parameters, aux_image, y, x, blocks):
    return get_distances(params, aux_image, y, x, blocks, mode="transfer")

def add_transfer_blocks(params: Parameters):
    temp_dim_block = params.dim_block
    temp_image = params.image.copy()
    result_image = np.zeros(params.dim_result_image)
    params.image = cv.resize(params.image, (params.dim_result_image[1], params.dim_result_image[0]))

    # for iteration in range(1, params.iterations + 1):
    for iteration in range(params.iterations):
        # alpha = 0.8 * (iteration / params.iterations) + 0.1
        alpha = 0.8 * (iteration / (params.iterations - 1)) + 0.1

        img_h = params.texture.shape[0]
        img_w = params.texture.shape[1]
        # generate random positions
        y = np.random.randint(low=0, high=img_h - params.dim_block, size=params.num_blocks)
        x = np.random.randint(low=0, high=img_w - params.dim_block, size=params.num_blocks)
        blocks = []
        for idx in range(params.num_blocks):
            pos_y = y[idx]
            pos_x = x[idx]
            blocks.append(params.texture[pos_y: pos_y + params.dim_block, pos_x: pos_x + params.dim_block])
        # blocks = np.array(blocks, np.float32)

        overlap = int(np.round(params.block_overlap * params.dim_block))
        new_dim_block = params.dim_block - overlap
        # print(overlap, new_dim_block)

        n_blocks_x = int(np.floor(params.image.shape[1] / new_dim_block))
        n_blocks_y = int(np.floor(params.image.shape[0] / new_dim_block))

        if params.image.ndim == 3:
            aux_image = np.zeros((n_blocks_y * new_dim_block + overlap, n_blocks_x * new_dim_block + overlap, 3), np.uint8)
        else:
            aux_image = np.zeros((n_blocks_y * new_dim_block + overlap, n_blocks_x * new_dim_block + overlap), np.uint8)

        params.image = cv.resize(temp_image,   (aux_image.shape[1], aux_image.shape[0]))
        result_image = cv.resize(result_image, (aux_image.shape[1], aux_image.shape[0]))

        # blur_dim = round(params.dim_block * 1.5)
        blur_dim = round(params.dim_block * 1)
        blur_dim = blur_dim + (1 - blur_dim % 2)
        params.image = cv.GaussianBlur(
            params.image,
            (blur_dim, blur_dim),
            0)

        # cv.imshow("a",params.image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        for y in range(n_blocks_y):
            for x in range(n_blocks_x):
                cv.imshow('img', aux_image)
                cv.waitKey(100)
                # cv.destroyAllWindows()
                print(" %.2f%%" % ((y * n_blocks_x + x) / (n_blocks_x * n_blocks_y) * 100), end='\r')

                intensities = get_intensities(params, params.image, y, x, blocks)
                intensities = np.array(intensities)
                prev_distances = np.zeros(params.num_blocks)
                if iteration > 0:
                    prev_distances = get_distances(params, result_image, y, x, blocks)
                distances = get_distances(params, aux_image, y, x, blocks)
                distances = np.array(distances)
                final_distances = alpha * (distances + prev_distances) + (1-alpha) * intensities

                min_dist = min(final_distances)
                if min_dist <= 0:
                    min_dist = 0.1
                max_dist = min_dist * (1 + params.tolerance)

                candidates = np.where(final_distances <= max_dist)[0]
                # print(alpha, min_dist, max_dist, len(candidates))
                idx = np.random.randint(low=0, high = len(candidates), size=1)[0]
                idx = candidates[idx]

                add_block_based_on_frontier(params, aux_image, y, x, blocks[idx])

        result_image = aux_image.copy()
        params.dim_block = params.dim_block // params.block_decrease

    result_image = cv.resize(result_image, (params.dim_result_image[1], params.dim_result_image[0]))
    params.dim_block = temp_dim_block
    params.image = temp_image.copy()
    return result_image



def create_image(params: Parameters):

    if params.method != "transfer":
        img_h = params.image.shape[0]
        img_w = params.image.shape[1]
        # generate random positions
        y = np.random.randint(low=0, high=img_h - params.dim_block, size=params.num_blocks)
        x = np.random.randint(low=0, high=img_w - params.dim_block, size=params.num_blocks)
        blocks = []
        for idx in range(params.num_blocks):
            pos_y = y[idx]
            pos_x = x[idx]
            blocks.append(params.image[pos_y: pos_y + params.dim_block, pos_x: pos_x + params.dim_block])
        # blocks = np.array(blocks, np.float32)

    result_image = None
    if params.method == "blocuriAleatoare":
        result_image = add_random_blocks(params, blocks)
    if params.method == "eroareSuprapunere":
        result_image = add_overlap_blocks(params, blocks)
    if params.method == "frontieraCostMinim":
        result_image = add_frontier_blocks(params, blocks)
    if params.method == "transfer":
        result_image = add_transfer_blocks(params)

    return result_image
