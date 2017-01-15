"""Grundfunktionalität für Aufgaben"""
import numpy as np
import glob
import scipy.misc


def get_dark_signal(sigma_dark_min, sigma_quantization_noise, system_gain):
    """

    :param sigma_dark_min: Varianz des temporären Rauschens.
                           Aufnahme mit minimaler Belichtungszeit bei verdeckter Kamera.
    :param sigma_quantization_noise: Varianz des Quantisierungsrauschen. Meist 1/12
    :param system_gain:
    :return:
    """
    if sigma_dark_min < 0.24:
        """
        Das temporäre Rauschen wird vom Quantisierungsrauschen dominiert.
        Setze sigma_dark_min auf 0.49. Siehe S.18, EMVA 1288, Release 3.1
        """
        sigma_dark_min = 0.49

    return (sigma_dark_min - sigma_quantization_noise) / (system_gain ** 2)


def get_minimum_irradiation(quantum_efficiency, variance_dark_noise, system_gain):
    """Berechnet die kleinste nutzbare Bestrahlungsstärke"""
    # Skript 2, S. 22
    minimum_irradiation = (1 / quantum_efficiency) * (variance_dark_noise / system_gain + 0.5)

    return minimum_irradiation


def get_time_variance(mat1, mat2):
    """Liefert zeitliche Varianz von zwei Matrizen zurück"""
    # Liefert Dimension der Matrix zurück
    height, width = mat1.shape
    total_dim = height * width * 2

    # Skript 2, S. 13
    variance = np.sum((mat1 - mat2) ** 2) / total_dim

    return variance


def get_time_variance_of_two_images(images):
    """
    Gibt die zeitliche Varianz von zwei aufeinander folgenden Bildern zurück.
    Bilder sollten entsprechend zusammenhängend sortiert sein.
    """
    counter = 1
    variance_list = []

    last_image = None

    for image in images:
        if counter == 1:
            last_image = image
            counter += 1
        else:
            variance = get_time_variance(image, last_image)
            variance_list.append(variance)
            counter = 1

    return variance_list


def get_mean_of_two_images(images):
    """
    Gibt den Mittelwert von zwei aufeinander folgenden Bilder zurück.
    Bilder sollten entsprechend zusammenhängend sortiert sein.
    """
    counter = 1
    mean_list = []

    last_image = None

    for image in images:
        if counter == 1:
            last_image = np.array(image)
            counter += 1
        else:
            two_images = np.append(last_image, image)
            mean = np.mean(two_images)
            mean_list.append(mean)
            counter = 1

    return mean_list


def interpolate_dark_image(dark1, t1, dark2, t2, t_new):
    if t_new > t2:
        raise RuntimeError("t_new should be <= t2")

    """Skript 3, S. 24"""
    dark_new = ((t2 - t_new) / (t2 - t1)) * dark1 + ((t_new - t1) / (t2 - t1)) * dark2

    return dark_new

def flat_field(image, dark_image, image_50):
    """Skript 3, S. 25"""
    return np.mean(image_50) * (image - dark_image) / image_50

def get_radiation_energy(area, irradiance, exposure_time):
    """Liefert die Strahlungsenergie"""
    # Skript 2, S. 5
    return area * irradiance * exposure_time


def interpolate_dead_pixels(image, dead_pixels):
    height, width = image.shape

    for y, x in zip(*dead_pixels):
        # Oben
        val1 = image[y - 1, x] if y - 1 > 0 else 0

        # Unten
        val2 = image[y + 1, x] if y + 1 < height else 0

        # Rechts
        val3 = image[y, x + 1] if x + 1 < width else 0

        # Links
        val4 = image[y, x - 1] if x - 1 > 0 else 0

        image[y, x] = (val1 + val2 + val3 + val4) / 4.0

def get_mean_of_photons(area, irradiance, exposure_time, wavelength):
    """
    Liefert die mittlere Anzahl an einfallenden Photonen
    auf der Sensorfläche
    :param area in m²
    :param irradiance in W/m²
    :param exposure time in seconds
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
        image = scipy.misc.imread(image_path, mode='L')
        sorted_images.append(image)

    return sorted_images
