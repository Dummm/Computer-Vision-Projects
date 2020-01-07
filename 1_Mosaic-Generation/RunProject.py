"""
    PROIECT MOZAIC
"""

"""
    Mail laboranta:
        mariana-iuliana.georgescu@my.fmi.unibuc.ro
        sau pe pdf
    Exercitii:
        - a) 'caroiaj' pentru imaginiTest cu dimensiunile (100, 75, 50, 25) distanta euclidiana dintre culori
        - b) 'aleator' ...
        - c) a) + difrenta intre piesele adiacente
        d) 5 mozaicuri tematice
        e) a) + hexagoane
        f) e) + difrenta intre piesele adiacente

    TODO:
        Avg. color folosind np.mean(..., axis=(1, 2))
        Fa imaginile cu np.float32(img)
        Greyscale (+ ndims)
        Tweak la random
        Pozitia in random folosind un singur numar (x*H + y)
        Hexagoane pe coloane
            impare
                y in range(14, img.shapep[0]-h, h)
                x in range(0, img.shape[1]-w, w + w // 3)
            pare
                y in range(0, img.shapep[0]-h, h)
                x in range(2/3*w, img.shape[1]-w, w + w // 3)

        BuildMosaic.py
            - load_pieces
            - compute_dimensions
        AddPiecesMosaic.py
            add_pieces_grid
            add_pieces_random
            add_pieces_hexagon
"""

# Parametrii algoritmului sunt definiti in clasa Parameters.
from Parameters import *
from BuildMosaic import *

"""
## Cod original

# numele imaginii care va fi transformata in mozaic
image_path = './../data/imaginiTest/ferrari.jpeg'
params = Parameters(image_path)

# directorul cu imagini folosite pentru realizarea mozaicului
params.small_images_dir = './../data/colectie/'
# tipul imaginilor din director
params.image_type = 'png'
# numarul de piese ale mozaicului pe orizontala
# pe verticala vor fi calcultate dinamic a.i sa se pastreze raportul
params.num_pieces_horizontal = 25
# afiseaza piesele de mozaic dupa citirea lor
params.show_small_images = False
# modul de aranjarea a pieselor mozaicului
# optiuni: 'aleator', 'caroiaj'
params.layout = 'aleator'
# params.layout = 'caroiaj'
# criteriul dupa care se realizeaza mozaicul
# optiuni: 'aleator', 'distantaCuloareMedie'
# params.criterion = 'aleator'
params.criterion = 'distantaCuloareMedie'
# daca params.layout == 'caroiaj', sa se foloseasca piese hexagonale
params.hexagon = False

img_mosaic = build_mosaic(params)
cv.imwrite('mozaic.png', img_mosaic)
"""


# Cod folosit pentru generarea imaginilor
images_path = './../data/imaginiTest'
colection_path = './../data/colectie/'
files = os.listdir(images_path)

cifar_images_path = './../data/cifar_imaginiTest'
cifar_colection_path = './../data/cifar_colectie/'
cifar_files = os.listdir(cifar_images_path)

mask_path = './../data/mask2.jpg'

mozaic_dimensions = [25, 50, 75, 100]
random_dimensions = [50]

# for file in cifar_files:
for file in files:
    # for dimension in random_dimensions:
    for dimension in mozaic_dimensions:
        image_path = os.path.join(images_path, file)
        # image_path = os.path.join(cifar_images_path, file)

        if file == files[0] or file == files[3]:
            params = Parameters(image_path, True)
        else:
            params = Parameters(image_path)

        params.small_images_dir = colection_path
        # params.small_images_dir = cifar_colection_path + file[:-4]

        params.image_type = 'png'
        params.num_pieces_horizontal = dimension
        params.show_small_images = False

        params.layout = 'caroiaj'
        # params.layout = 'aleator'

        # params.criterion = 'aleator'
        # params.criterion = 'distantaCuloareMedie'
        params.criterion = 'distantaCuloareMedie2'

        params.hexagon = True
        # params.hexagon = False
        params.hex_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        null, params.hex_mask = cv.threshold(params.hex_mask, 127, 255, cv.THRESH_BINARY)

        # filename = params.layout + '_' + file.split('.')[0] + "_" + str(dimension) + ".png"
        # filename = "cifar_" + params.layout + '_' + file.split('.')[0] + "_" + str(dimension) + ".png"
        # filename = params.layout + '_dist_' + file.split('.')[0] + "_" + str(dimension) + ".png"
        # filename = params.layout + '_hex_' + file.split('.')[0] + "_" + str(dimension) + ".png"
        filename = params.layout + '_hex_dist_' + file.split('.')[0] + "_" + str(dimension) + ".png"

        if filename not in os.listdir('./'):
            debug('Building ' + filename)
            img_mosaic = build_mosaic(params)
            cv.imwrite(filename, img_mosaic)


"""
# Cod folosit pentru scos imaginile din dataset-uri
datasets_path = './../data/cifar-10-batches-py'
datasets = os.listdir(datasets_path)
for dataset in datasets:
    datadict = unpickle(datasets_path + "/" + dataset)
    X = datadict["data"]
    Y = datadict['labels']
    Z = datadict["filenames"]
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)

#     for label in range(10):
#         os.mkdir(datasets_path + "/" + str(label))
    label_count = np.zeros(10)
    for i in range(len(X)):
        if(label_count[Y[i]] < 500):
            filename = datasets_path + "/" + str(Y[i]) + '/' + Z[i]
            label_count[Y[i]] += 1
            cv.imwrite(filename, X[i])
"""