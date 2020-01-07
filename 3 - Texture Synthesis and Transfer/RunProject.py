"""
    PROIECT
    Sinteza si transferul texturii
    Implementarea a proiectului Sinteza si transferul texturii
    dupa articolul "Seam Carving for Content-Aware Image Resizing", autori Alexei A. Efros si William T. Freeman
"""

"""
    NOTES
    Info
        max_dist e ca sa fie diferite rezultatele
        tol_error ca sa nu dea erori de la 0
        incearca cu masti, ba baiatule
        la textura incearca sa faci distanta pe imaginile grayscale
        la fiecare subpunct folosesti codu de la subpunctu precedent
        alpha trebuie setat de tine (0.1)
        incearca sa faci block-urile puteri de 2 sau 3
    Rezolvare
        Generam 2000 de blocuri 36x36
        Primul bloc e aleator
        Urmatoarele in functie distanta din overlap
            Prima linie doar pe vertical
            Urmatoarele si pe orizontal
        Generare/adaugare block-uri
            n_blocks_x = ...
            n_blocks_y = ...
            aux_image = np.zeros((...))
            for y in range(n_blocks_y):
                for x in range(n_blocks_x):
                    distance_x = zeros num_blocks
                    distance_y = zeros num_blocks
                    distance_x_y = zeros num_blocks

                    start_y = y * (dim_block_overlap)
                    end_y = start_y + dim_block
                    start_x = ...
                    end_x = ...
                    if x == 0 and y == 0:
                        idx_random
                    if x > 0:
                        distances_x = get_distances(
                            blocks, aux_image[start_y:end_y, start_x:end_x], overlap, 'x')
                    if y > 0:
                        ...
                    if x > 0 and y > 0:
                        ...
                    distances = distances_x + distances_y - distances_xy
                    min_dist = np.min(distances)
                    if min_dist == 0:
                        min_dist = 0.1 # Ceva mic != 0
                    max_dist = (1 + tol_error) + min_dist # Cauta 'tolerance' in pdf
                    candidates = np.where(distances <= max_dist)[0]
                    random_idx = np.random.randint(low=0, high=len(candidates), ???=1)[0]
                    patch_idx = camdidates[random_idx]
        Calc dist.
                def get_distances(blocks, patch, overlap, mode)
                    if mode == 'x':
                        small_blocks = blocks[:, :, :overlap] # 2000x36x6x3
                        small_patch = patch[:, :overlap] # 36x6x3
                    if mode == 'y':
                        small_blocks = blocks[:, :overlap]
                        small_patch = patch[:overlap]
                    if mode == 'xy':
                        small_blocks = blocks[:, :overlap, :overlap]
                        small_patch = patch[:overlap, :overlap]

                    distances = np.sum((small_blocks - small_patch) ** 2, axis=(1, 2, 3))
        Frontiere
            Drum minim pe overlap
                Generezi energie pe cele doua patch-uri
                Energia = Diferenta la patrat dintre ele
                    La un moment dat faci suma pe canale
            Folosesti masca cu 1 si 0
                pe patch-ul orizontal il rotesti pentru drum
            sedocod
                aux_image[start_y:end_y, start_x:end_x] =
                    (1-M) * aux_image[start_y:end_y, start_x:end_x] + M * blocks[patch_idx]
        Textura
            # seidockod
            result_image = np.zeros src_image.shape
            for it in iters:
                generare blocuri
                n_blocks_x = ...
                n_blocks_y = ...
                aux_image = zeros...
                for y in range(n_blocks_y):
                    for x in range(n_blocks_x):
                        start_x, start_y, end_x, end_y ...
                        src_patch = src_image[start_y:end_y, ...] # Patch-ul care trebuie inlocuit
                        intensity = get_distances(blocks, source_patch, 0, mode='source')
                        distances = ca in vechime
                        # new_d = alpha * distances + (1 - alpha) * intensity
                        # new_d = alpha * (distances + prev_distances) + (1 - alpha) * intensity
                result_image = aux_image.sopy()

                # undeva, nu stiu unde
                if it > 0:
                    patch_prev = result_image[start_y: end_y,...]
                    prev_distances = get_distances(blocks, patch_prev, 0, 'source')

    NOTES II
    Pune toleranta, baet
        tol_err = 0.1 # default
        Memorezi distantele pentru toate block-urile in vector
        Iei distanta minima si calculezi distanta maxima
            max_dist = min_dist + (1 + tol_err)
        Daaca min_dist == 0, min_dist = 0.1 sau ceva
        candidates = np.where(distances <= max_dist)[0] # [0] nu il ia pe primu, e susta
        idx = np.randint(low=0, high=len(candidates), size=1)[0] # same
        aux_image[...] = candidates[idx]
    Incearca matricea de energie la frontiera minima
        E = np.sum((patch1 - patch2)**2, axis = 2)
    Transfer
        result_image = zeros
        # o sa se schimbe dimensiunea la result_image
        for iter in range(num_iter):
            *generare blocuri*
            n_blocks_s, n_blocks_y = ...
            aux_image = ...
            for y in range(n_blocks_y):
                for x in range(n_blocks_x):
                    start_x, end_x, start_y, end_y = ...
                    patch_source = params.source_image[start_y: end_y, start_x:end_x]
                    intensity = get_distances(blocks, patch_source, 0, mode='source') # source = distanta pe tot blocu
                    prev_distances = zeros 2000
                    if it > 0:
                        prev_patch = result_image[start_y:end_y...]
                        prev_distances = get_distances(blocks, prev_patch, 0, mode='source')
                    distances = distances_x + distances_y + distances_x_y
                    all_distances = alpha * (distances + prev_distances) + (1-alpha) * intensity
            params.dim_block = params.dim_block // 3 # sau N idk
            alpha = 0.8 * (iter - 1) / (num_iter - 1)+ 0.1
            result_image = aux_image.copy()


"""

from Parameters import *
from CreateImage import *

# image_name = '../data/radishes.jpg'
# image_name = '../data/brick.jpg'
# image_name = '../data/rice.jpg'
# image_name = '../data/img1.png'
# image_name = '../data/img2.png'
# image_name = '../data/img3.png'
# image_name = '../data/img4.png'
# image_name = '../data/img5.png'
# image_name = '../data/img6.png'
# image_name = '../data/img7.png'
# image_name = '../data/img8.png'
# image_name = '../data/img9.png'
# image_name = '../data/img10.png'
# image_name = '../data/img11.png'
# image_name = '../data/eminescu.jpg'
# image_name = '../data/am.jpg'
image_name = '../data/dali.jpg'
params:Parameters = Parameters(image_name)

# dimensiunea imaginii ce urmeaza a fi construita
image_resize_ratio = 1
# image_resize_ratio = 2
params.dim_result_image = (image_resize_ratio * params.image.shape[0], image_resize_ratio * params.image.shape[1])

# numarul de blocuri ce vor fi extrase din imaginea originala
# params.num_blocks = 2000
# params.num_blocks = 5000
params.num_blocks = 10000
# params.num_blocks = 50000

# dimensiunea blocurilor
# params.dim_block = 25 * 9
params.dim_block = 20 * 9
# params.dim_block = 10 * 9

# suprapunerea blocurilor
# params.block_overlap = 1 / 6
params.block_overlap = 1 / 3
# print(params.block_overlap)

# metoda de aranjare a blocurilor
# blocuriAleatoare, eroareSuprapunere, frontieraCostMinim, transfer
# params.method = 'blocuriAleatoare'
# params.method = 'eroareSuprapunere'
# params.method = 'frontieraCostMinim'
params.method = 'transfer'

# replacement texture
# texture_name = '../data/rice.jpg'
# texture_name = '../data/radishes.jpg'
# texture_name = '../data/toast3.png'
# texture_name = '../data/img1.png'
texture_name = '../data/the-persistence-of-memory-1931.jpg'
# texture_name = '../data/gm.jpg'
params.texture = cv.imread(texture_name)

params.iterations = 3
params.block_decrease = 3

# '''
print('Generating image...')
img = create_image(params)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

from datetime import datetime
filename = image_name[image_name.rfind('/') + 1:image_name.rfind('.')]
filename = params.method + '_' + filename + '_' + str(datetime.now().date()) + '_' + str(datetime.now().time().strftime("%H%M%S%f")) + '.jpg'
print('Writing image \'' + filename + '\'...')
cv.imwrite('../Imagini/' + filename, img)
# '''

'''
import os
methods = ['blocuriAleatoare', 'eroareSuprapunere', 'frontieraCostMinim']
images_path = '../data'
save_path = '../Imagini/[a] Sinteza texturii'
try:
    os.mkdir(save_path)
except OSError as error:
    print(error)
files = os.listdir(images_path)
ext = '.jpg'

for file in files:
    if file != 'img5.png':
        continue
    dirName = os.path.join(save_path, file)
    try:
        os.mkdir(dirName)
    except OSError as error:
        print(error)

    image_path = os.path.join(images_path, file)
    name = image_path[image_path.rfind('/') + 1:image_path.rfind('.')]
    print(file)

    params = Parameters(image_path)
    image_resize_ratio = 2
    params.dim_result_image = (image_resize_ratio * params.image.shape[0], image_resize_ratio * params.image.shape[1])

    # params.num_blocks = 2000
    # params.num_blocks = 5000
    params.num_blocks = 10000
    # params.num_blocks = 50000

    params.dim_block = 36
    params.block_overlap = 1 / 6

    cv.imwrite(dirName + '/1_' + name + '_original'      + ext, params.image)
    for i, method in enumerate(methods, 2):
        print('\t' + method)
        params.method = method
        img = create_image(params)
        cv.imwrite(dirName + '/' + str(i) + '_' + name + '_' + method + ext, img)
        print(end='\r\n')
'''
