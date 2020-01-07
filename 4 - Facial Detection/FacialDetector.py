from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None

    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        D = (self.params.dim_window/self.params.dim_hog_cell - (self.params.cells_per_block - 1)) ** 2 * self.params.dim_descriptor_cell
        positive_descriptors = []
        print('Calculam descriptorii pentru %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i, end='\r')
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            img = cv.equalizeHist(img)
            descriptor, hog_img = hog(
                img,
                pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                feature_vector=True,
                visualize=True
            )
            # print(D, descriptor.shape)
            # cv.imshow('da', img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # positive_descriptors.append(descriptor)

            if self.params.use_flip_images:
                img = cv.flip(img, 1)
                descriptor, hog_img = hog(
                    img,
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                    feature_vector=True,
                    visualize=True
                )
                positive_descriptors.append(descriptor)
            '''
            if self.params.use_contrasted_images:
                img = cv.equalizeHist(img)
                descriptor, hog_img = hog(
                    img,
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                    feature_vector=True,
                    visualize=True
                )

                if self.params.use_flip_images:
                    img = cv.flip(img, 1)
                    descriptor, hog_img = hog(
                        img,
                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                        feature_vector=True,
                        visualize=True
                    )
                    positive_descriptors.append(descriptor)
            '''

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        D = (self.params.dim_window/self.params.dim_hog_cell - (self.params.cells_per_block - 1)) ** 2 * self.params.dim_descriptor_cell
        negative_descriptors = []
        print('Calculam descriptorii pentru %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i, end='\r')
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            for j in range(num_negative_per_image):
                start_y = np.random.randint(low=0, high=img.shape[0] - self.params.dim_window)
                start_x = np.random.randint(low=0, high=img.shape[1] - self.params.dim_window)
                end_y = start_y + self.params.dim_window
                end_x = start_x + self.params.dim_window
                descriptor, hog_img = hog(
                    img[start_y:end_y, start_x:end_x],
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                    feature_vector=True,
                    visualize=True
                )
                # print(D, descriptor.shape)
                # cv.imshow('da', hog_img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                negative_descriptors.append(descriptor)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
            (self.params.dim_hog_cell,
            self.params.number_negative_examples,
            self.params.number_positive_examples)
        )
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]
        print(negative_scores[:5])
        print(positive_scores[:5])


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        # print(bbox_a, bbox_b)
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        # print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self, return_descriptors=False):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        # detections = None  # array cu toate detectiile pe care le obtinem
        detections = np.array([])  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le optinem
        file_names = np.array([])  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            file_name = test_files[i][test_files[i].rfind('/') + 1:]
            # file_name = test_files[i][test_files[i].rfind('\\') + 1:]
            print('Procesam imaginea de testare %d/%d [%s]..' % (i + 1, num_test_images, file_name))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            # img = cv.equalizeHist(img)
            POWERFUL_negative_descriptors = []

            img_temp = img.copy()
            image_detections = []
            image_scores = []
            image_shape = img_temp.shape
            dim_block = int(self.params.dim_window / self.params.dim_hog_cell - (self.params.cells_per_block - 1))


            num_scaling = 0
            scale = 1.0
            while True:
                if num_scaling > 0:
                    # img = cv.resize(img, (img.shape[1] * self.params.scaling_ratio, img.shape[0] * self.params.scaling_ratio))
                    img = cv.resize(img, dsize=None, fx=self.params.scaling_ratio, fy=self.params.scaling_ratio)
                    scale = scale * self.params.scaling_ratio
                    if min(img.shape[0], img.shape[1]) < self.params.dim_window:
                        break
                num_scaling = num_scaling + 1

                # cv.imshow('??', img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                descriptor, hog_img = hog(
                    img,
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(self.params.cells_per_block, self.params.cells_per_block),
                    feature_vector=False,
                    visualize=True
                )
                # print(descriptor.shape)
                # print(self.params.dim_hog_cell, self.params.cells_per_block)

                for y in range(descriptor.shape[0] - (dim_block - 1)):
                    for x in range(descriptor.shape[1] - (dim_block - 1)):
                        block_descriptor = descriptor[y:y+dim_block, x:x+dim_block]
                        block_descriptor = np.ravel(block_descriptor)

                        rez = self.best_model.decision_function([block_descriptor])[0]
                        # rez = np.dot(block_descriptor, w) + bias

                        if rez > 0 and return_descriptors:
                            POWERFUL_negative_descriptors.append(block_descriptor)
                            continue

                        if rez > self.params.threshold:
                            x_min = int(x * self.params.dim_hog_cell * (1 / scale))
                            y_min = int(y * self.params.dim_hog_cell * (1 / scale))
                            x_max = int(x_min + self.params.dim_window * (1 / scale))
                            y_max = int(y_min + self.params.dim_window * (1 / scale))
                            # print(rez)
                            # print(y_min, y_max, x_min, x_max)
                            # cv.imshow('??', img_temp[y_min:y_max, x_min:x_max])
                            # cv.waitKey(0)
                            # cv.destroyAllWindows()
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(rez)

            if not return_descriptors:
                image_detections = np.array(image_detections)
                image_scores = np.array(image_scores)
                if len(image_detections) != 0:
                    image_detections, image_scores = self.non_maximal_suppression(image_detections, image_scores, image_shape)
                # img = img_temp.copy()
                # for (i, rect) in enumerate(image_detections):
                #     print(image_scores[i])
                #     cv.imshow('??', img_temp[rect[1]:rect[3], rect[0]:rect[2]])
                #     cv.waitKey(0)
                #     cv.destroyAllWindows()
                #     cv.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 1)
                # cv.imshow('??', img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                img = img_temp.copy()
                image_file_names = np.array([file_name for x in range(len(image_detections))])

                if len(image_detections) != 0:
                    if len(detections) != 0:
                        detections = np.concatenate((detections, image_detections))
                        scores     = np.concatenate((scores, image_scores))
                        file_names = np.concatenate((file_names, image_file_names))
                    else:
                        detections = image_detections.copy()
                        scores = image_scores.copy()
                        file_names = image_file_names.copy()

            # detections = np.append(detections, image_detections)
            # scores     = np.append(scores, image_scores)
            # file_names = np.append(file_names, image_file_names)

            # print(detections)
            # print(scores)
            # print(file_names)

            # for i in range(len(image_detections)):
            #     detections = np.append(detections, image_detections[i])
            #     scores = np.append(scores, image_scores[i])
            #     file_names = np.append(file_names, image_file_names[i])


            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.\n'
                  % (i + 1, num_test_images, end_time - start_time))
        if return_descriptors:
            return POWERFUL_negative_descriptors
        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) -  1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
