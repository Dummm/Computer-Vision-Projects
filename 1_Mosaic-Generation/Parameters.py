import cv2 as cv

# dibag
DEBUG = True
def debug(*args):
    if DEBUG:
        print('\033[92m', end = '')
        print(*args)
        print('\033[0m', end = '')

# CIFAR dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
class Parameters:

    def __init__(self, image_path, grayscale = False):
        self.image_path = image_path
        if grayscale:
            self.image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        else:
            self.image = cv.imread(image_path)
        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)

        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None

        # Parametri adaugati
        self.grayscale = grayscale
        self.small_images_colors = None
        self.hex_mask = None
        self.small_images_colors_hex = None