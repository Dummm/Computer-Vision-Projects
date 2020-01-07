import cv2 as cv
import sys


class Parameters:
    def __init__(self, image_name):
        self.image_name = image_name
        self.image = cv.imread(image_name)
        if self.image is None:
            print('The image name %s is invalid.' % self.image_name)
            sys.exit(-1)
        self.dim_result_image = (2 * self.image.shape[0], 2 * self.image.shape[1])
        self.num_blocks = 2000
        self.dim_block = 36
        self.method = 'blocuriAleatoare'

        # Added parameters
        self.block_overlap = 1 / 6
        self.tolerance = 0.1
        self.texture = None
        self.iterations = 3
        self.block_decrease = 3

