import libcore
import matplotlib.pyplot as plt
import os
import numpy as np

"""Berechnete Bestrahlungsstärke E. Siehe Rechnung in Ordner MessungenAufgabe_1-2"""
IRRADIANCE_WATT_PER_SQUARE_METER = 0.121705797795879

"""Wellenlänge der verwendeten grünen LED"""
WAVELENGTH_METER = 0.000000525

"""Pixelgröße aus Datenblatt entnommen"""
PIXEL_AREA_METER = 0.0000045 ** 2

TIME_OF_EXPOSURE_MS = [0.02,
                       1.645,
                       3.27,
                       4.895,
                       6.52,
                       8.145,
                       9.77,
                       11.395,
                       13.02]


def plot_mean_of_photons():
    mean_of_photons_for_texp = []

    for texp in TIME_OF_EXPOSURE_MS:
        texp_sec = texp / 1000
        mean_of_photons = libcore.get_mean_of_photons(PIXEL_AREA_METER,
                                            IRRADIANCE_WATT_PER_SQUARE_METER,
                                            texp_sec,
                                            WAVELENGTH_METER)

        mean_of_photons_for_texp.append(mean_of_photons)

    plt.plot(TIME_OF_EXPOSURE_MS, mean_of_photons_for_texp, 'ro')
    plt.xlabel('Time of Exposure [ms]')
    plt.ylabel('Mean of Photons per Pixel')

    plt.show()


def get_mean_gray_value_without_dark_noise():
    # ../MessungenAufgabe_1-2/offen
    path_open = os.path.join(os.path.join("..", "MessungenAufgabe_1-2"), "offen/*")

    # ../MessungenAufgabe_1-2/geschlossen
    path_closed = os.path.join(os.path.join("..", "MessungenAufgabe_1-2"), "geschlossen/*")

    # Verarbeite images_open
    images_open = libcore.get_sorted_images(path_open)

    counter = 1
    mean_open = []

    last_image = None

    for image_open in images_open:
        if counter == 1:
            last_image = image_open
            counter += 1
        else:
            mean = libcore.get_mean([image_open, last_image])
            mean_open.append(mean)
            counter = 1

    mean_open = np.matrix(mean_open)

    # Verarbeite images_closed
    images_closed = libcore.get_sorted_images(path_closed)
    counter = 1
    mean_closed = []

    last_image = None

    for image_closed in images_closed:
        if counter == 1:
            last_image = image_closed
            counter += 1
        else:
            mean = libcore.get_mean([image_closed, last_image])
            mean_closed.append(mean)
            counter = 1

    mean_closed = np.matrix(mean_closed)

    mean_gray_value_without_dark_noise = mean_open - mean_closed

    return mean_gray_value_without_dark_noise

def main():
    #plot_mean_of_photons()
    get_mean_gray_value_without_dark_noise()

if __name__ == '__main__':
    main()
