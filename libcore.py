"""Grundfunktionalität für Aufgaben"""
import numpy as np


def get_time_variance(mat1, mat2):
    """Liefert zeitliche Varianz von zwei Matrizen zurück"""
    # Liefert Dimension der Matrix zurück
    height, width = mat1.shape
    total_dim = height * width * 2

    variance = np.sum((mat1 - mat2) ** 2) / total_dim

    return variance


def test_variance():
    matrix1 = np.ones((5, 5))
    matrix2 = np.ones((5, 5)) * 2

    print(get_time_variance(matrix1, matrix2))


test_variance()