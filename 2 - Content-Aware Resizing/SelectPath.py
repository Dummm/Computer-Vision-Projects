import sys
import numpy as np
import matplotlib.pyplot as plt


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    pathway = [(line, col)]
    for i in range(1, E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if pathway[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif pathway[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = pathway[-1][1] + opt
        pathway.append((line, col))

    return pathway


def select_greedy_path(E):
    line = 0
    column = np.argmin(E[line, :])
    pathway = [(line, column)]
    for i in range(1, E.shape[0]):
        line = i
        column = pathway[-1][1]
        if column == 0:
            opt = np.argmin(E[line,column:column+2])
        elif column == E.shape[1]-1:
            opt = np.argmin(E[line,column-1:column+1]) - 1
        else:
            opt = np.argmin(E[line,column-1:column+2]) - 1
        pathway.append((line, column+opt))

    return pathway


def select_dynamic_path(E):
    # Matricea de distante
    D = np.ndarray(E.shape, dtype=E.dtype)
    D[0, :] = E[0, :]
    for i in range(1, D.shape[0]):
        D[i, 0] = E[i, 0] + min(D[i-1, 0], D[i-1, 1])
        for j in range(1, D.shape[1] - 1):
            D[i, j] = E[i, j] + min(D[i-1, j-1], D[i-1, j], D[i-1, j+1])
        D[i, D.shape[1] - 1] = E[i, D.shape[1] - 1] + min(D[i-1, D.shape[1] - 2], D[i-1, D.shape[1] - 1])

    # Crearea drumului
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

    # plt.imshow(D)
    # x, y = zip(*pathway)
    # plt.scatter(y, x)
    # plt.show()

    return pathway


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'greedy':
        return select_greedy_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)