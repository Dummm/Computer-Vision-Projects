"""
    PROIECT
    REDIMENSIONEAZA IMAGINI.
    Implementarea a proiectului Redimensionare imagini
    dupa articolul "Seam Carving for Content-Aware Image Resizing", autori S. Avidan si A. Shamir
"""

"""
    Progress
        -- 1.1 Micşorarea imaginii pe lăţime (decrease_width)
            -- compute energy - funcţia trebuie completată de către voi folosind ecuaţia (1) din articolul ataşat
            -- SelectPath.py - scriptul este scris parţial, trebuie să-l completaţi voi
            -- delete_path - funcţia este scrisă parţial, trebuie să o completaţi voi
        -- 1.2 Micşorarea imaginii pe ı̂nălţime
            -- decrease_height
        -- 1.3 Mărirea imaginilor
            -- increase_width
            -- increase_height
        -- 1.4 Amplificarea conţinutului imaginilor
            -- amplify_content
        1.5 Eliminarea unui obiect din imagine
            delete_object
        1.6 Predarea proiectului
            PDF
                (a) 1.1 - castel.jpg
                (b) 1.2 - praga.jpg
                (c) 1.3 - delfin.jpg
                (d) 1.4 - arcTriumf.jpg
                (e) 1.5 - lac.jpg
                (f) 5 imagini, toate metodele, cel putin 3 exemple reusite si 2 nereusite + exemplificari
    Notes
        peturi
        Float-uri pentru imagini, uint8 pentru afisari si alte chestii
        Filtru sobel
        Ca sa micsorezi pe cealalta directie, rotesti imaginea
            np.rot(img, k = 3) <- de k ori?
        Resize
            resized_img = cv.resize(img, (0, 0), fx = factor, fy = factor)
            pixel_w = resized_img.shape[1] - img.shape[1]
            pixel_w = resized_img.shape[0] - img.shape[0]
        Stergerea unui obiect
            Modifici E, pui -1000 pe obiect
            x0, y0, w, h = cv.selectROI(img)
            xmin = x0
            xmax = x0 + w
            ymin = y0
            ymax = y0 + h
            Scazi din E folosind ^
                Nu uita sa modifici max-uri si sa rotesti E, idk
        Marirea imaginii
            img_copie = img.copy()
            for i in range(50):
                selectezi path
                il salvezi
                il stergi
                (totu pe copie)
            cand adaugi, pe o alta copie btw, c primeste media
                dintre c si c-1, si c+1 media dintre c+1 si c+2 poate
            sau medie pe c+1, dintre c-1 is c+2 (poate si cu c)
            La un moment dat, adaugi 2 la path-uri(prrobabil de fiecare data cand scoti unu din lista)
                Path-urile de la dreapta
"""

from Resize_Image import *
import matplotlib.pyplot as plt

# image_name = '../data/castel.jpg'
# image_name = '../data/praga.jpg'
image_name = '../data/delfin.jpeg'
# image_name = '../data/arcTriumf.jpg'
# image_name = '../data/lac.jpg'
params = Parameters(image_name)

# seteaza optiunea de redimenionare
# micsoreazaLatime, micsoreazaInaltime, maresteLatime, maresteInaltime, amplificaContinut, eliminaObiect
# params.resize_option = 'micsoreazaLatime'
# params.resize_option = 'micsoreazaInaltime'
# params.resize_option = 'maresteLatime'
params.resize_option = 'maresteInaltime'
# params.resize_option = 'amplificaContinut'
# params.resize_option = 'eliminaObiect'

# numarul de pixeli pe latime
params.num_pixels_width = 50

# numarul de pixeli pe inaltime
params.num_pixels_height = 50
# params.num_pixels_height = 100

# afiseaza drumul eliminat
params.show_path = True

# Scalarul de amplificare a continutului
params.amplification_factor = 1.5

# metoda pentru alegerea drumului
# aleator, greedy, programareDinamica
# params.method_select_path = 'aleator'
# params.method_select_path = 'greedy'
params.method_select_path = 'programareDinamica'

resized_image = resize_image(params)
resized_image_opencv = cv.resize(params.image, (resized_image.shape[1], resized_image.shape[0]))

cv.imwrite('d.jpg', resized_image)

'''
f, axs = plt.subplots(2, 2, figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.imshow(params.image[:, :, [2, 1, 0]])
plt.xlabel('original')

plt.subplot(1, 3, 2)
plt.imshow(resized_image_opencv[:, :, [2, 1, 0]])
plt.xlabel('OpenCV')

plt.subplot(1, 3, 3)
plt.imshow(resized_image[:, :, [2, 1, 0]])
plt.xlabel('My result')
plt.show()
'''

# [a..e]
'''
import os
name = image_name[image_name.rfind('/') + 1:image_name.rfind('.')]
ext = '.jpg'
# dirName = '[a] Micsorarea imaginii pe latime - ' + name + ext
# dirName = '[b] Micsorarea imaginii pe inaltime - ' + name + ext
# dirName = '[c] Marirea imaginilor - ' + name + ext
dirName = '[d] Amplificarea continutului imaginilor - ' + name + ext
# dirName = '[e] Eliminarea unui obiect din imagine - ' + name + ext
try:
    os.mkdir(dirName)
except OSError as error:
    print(error)
cv.imwrite(dirName + '/1_' + name + '_original'      + ext, params.image)
cv.imwrite(dirName + '/2_' + name + '_opencv'        + ext, resized_image_opencv)
cv.imwrite(dirName + '/3_' + name + '_content-aware' + ext, resized_image)
'''

# [f]
'''
import os
methods = ['aleator', 'greedy', 'programareDinamica']
images_path = '[f]'
files = os.listdir(images_path + '/originals')
ext = '.jpg'

for file in files:
    dirName = os.path.join(images_path, file)
    try:
        os.mkdir(dirName)
    except OSError as error:
        print(error)

    image_path = os.path.join(images_path + '/originals', file)
    name = image_path[image_path.rfind('/') + 1:image_path.rfind('.')]
    print(file)
    params = Parameters(image_path)
    params.resize_option = 'micsoreazaLatime'
    params.num_pixels_width = params.image.shape[1] // 4

    resized_image_opencv = cv.resize(params.image, (params.image.shape[1] - params.num_pixels_width, params.image.shape[0]))
    cv.imwrite(dirName + '/1_' + name + '_original'      + ext, params.image)
    cv.imwrite(dirName + '/2_' + name + '_opencv'        + ext, resized_image_opencv)

    for i, method in enumerate(methods, 3):
        print('\t' + method)
        params.method_select_path = method
        resized_image = resize_image(params)
        cv.imwrite(dirName + '/' + str(i) + '_' + name + '_content-aware_' + method + ext, resized_image)
        print(end='\r\n')
'''
