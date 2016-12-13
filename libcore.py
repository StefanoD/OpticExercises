"""Grundfunktionalität für Aufgaben"""
import numpy as np


def get_time_variance(mat1, mat2):
    """Liefert zeitliche Varianz von zwei Matrizen zurück"""
    # Liefert Dimension der Matrix zurück
    height, width = mat1.shape
    total_dim = height * width * 2

    # Skript 2, S. 13
    variance = np.sum((mat1 - mat2) ** 2) / total_dim

    return variance


def get_mean(mats):
    """Liefert den Mittelwert einer Matrix-Zelle"""
    total_mean = 0.0

    for mat in mats:
        total_mean += np.mean(mat)

    total_mean /= len(mats)

    return total_mean


def test_variance():
    matrix1 = np.ones((5, 5))
    matrix2 = np.ones((5, 5)) * 2

    print("time variance: ", get_time_variance(matrix1, matrix2))


def test_mean():
    matrix1 = np.ones((5, 5))
    matrix2 = np.ones((5, 5)) * 2

    print("Mean: ", get_mean([matrix1, matrix2]))


test_variance()
test_mean()