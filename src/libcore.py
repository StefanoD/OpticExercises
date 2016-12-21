"""Grundfunktionalität für Aufgaben"""
import numpy as np
import matplotlib.pyplot as plt
import glob


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


def get_radiation_energy(area, irradiance, exposure_time):
    """Liefert die Strahlungsenergie"""
    # Skript 2, S. 5
    return area * irradiance * exposure_time


def get_mean_of_photons(area, irradiance, exposure_time, wavelength):
    """
    Liefert die mittlere Anzahl an einfallenden Photonen
    auf der Sensorfläche
    :param area in m²
    :param irradiance in W/m²
    :param exposure_time in seconds
    :param wavelength in m
    """
    # Skript 2, S. 5
    hc = 5.034 * 10 ** 24

    return hc * area * irradiance * exposure_time * wavelength


def get_sorted_images(path):
    sorted_images = []

    # Aufsteigend sortierte Bild-Pfade
    image_paths = sorted(glob.glob(path))

    for image_path in image_paths:
        image = plt.imread(image_path)
        sorted_images.append(image)

    return sorted_images
