import libcore
import matplotlib.pyplot as plt
import numpy as np

from os.path import join
from scipy import stats

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
    # ../MessungenAufgabe_1-2/offen/*
    path_open = join(join(join("..", "MessungenAufgabe_1-2"), "offen", "*"))

    # ../MessungenAufgabe_1-2/geschlossen/*
    path_closed = join(join(join("..", "MessungenAufgabe_1-2"), "geschlossen"), "*")

    # Verarbeite images_open
    images_open = libcore.get_sorted_images(path_open)

    mean_open = libcore.get_mean_of_two_images(images_open)
    mean_open = np.array(mean_open)

    # Verarbeite images_closed
    images_closed = libcore.get_sorted_images(path_closed)

    mean_closed = libcore.get_mean_of_two_images(images_closed)
    mean_closed = np.array(mean_closed)

    mean_gray_value_without_dark_noise = mean_open - mean_closed

    return mean_gray_value_without_dark_noise


def get_variance_gray_value_without_dark_noise():
    # ../MessungenAufgabe_1-2/offen/*
    path_open = join(join(join("..", "MessungenAufgabe_1-2"), "offen", "*"))

    # ../MessungenAufgabe_1-2/geschlossen/*
    path_closed = join(join(join("..", "MessungenAufgabe_1-2"), "geschlossen"), "*")

    # Verarbeite images_open
    images_open = libcore.get_sorted_images(path_open)

    variance_open = libcore.get_time_variance_of_two_images(images_open)
    variance_open = np.array(variance_open)

    # Verarbeite images_closed
    images_closed = libcore.get_sorted_images(path_closed)

    variance_closed = libcore.get_time_variance_of_two_images(images_closed)
    variance_closed = np.array(variance_closed)

    variance_gray_value_without_dark_noise = variance_open - variance_closed

    return variance_gray_value_without_dark_noise


def plot_photo_transfer():
    mean_gray_value_without_dark_noise = get_mean_gray_value_without_dark_noise()
    variance_gray_value_without_dark_noise = get_variance_gray_value_without_dark_noise()

    # Limits der Axen setzen
    x_begin = 0
    x_end = 300
    plt.xlim(x_begin, x_end)
    plt.ylim(-1, np.max(variance_gray_value_without_dark_noise) * 1.3)

    # Plotte mittlere Grauwerte und Varianz ohne Dunkelstrom
    plt.plot(mean_gray_value_without_dark_noise, variance_gray_value_without_dark_noise, 'ro')
    plt.xlabel('gray value - dark value')
    plt.ylabel('variance gray value')

    # Sättingspunkt ist an der Stelle der maximalen Varianz
    saturation_index = np.argmax(variance_gray_value_without_dark_noise)

    # Sättingungspunkt im Plot einzeichnen
    saturation_x_coord = mean_gray_value_without_dark_noise[saturation_index]
    saturation_y_coord = variance_gray_value_without_dark_noise[saturation_index]

    plt.annotate('Saturation', xy=(saturation_x_coord, saturation_y_coord),
                xycoords='data',
                xytext=(-30, +30),
                textcoords='offset points',
                fontsize=16,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.plot([saturation_x_coord, saturation_x_coord], [-1, saturation_y_coord],
             color='red', linewidth=1.5, linestyle="--")

    # Lineare Regeression der steigenden Varianz bilden.
    # Die Sättigung soll dabei nicht miteinfließen.
    # Man geht davon aus, dass bei 70% des Sättigungspunktes die Sättigung anfängt
    mean_sat_begin = mean_gray_value_without_dark_noise[saturation_index] * 0.7

    # Ermittle die Indizes der Matrix-Einträge die einen kleineren Wert als mean_sat_begin haben
    mean_without_sat_indices = mean_gray_value_without_dark_noise < mean_sat_begin

    # Mean-Matrix von der Sättigung bereinigt
    mean_without_sat = mean_gray_value_without_dark_noise[mean_without_sat_indices]
    variance_without_sat = variance_gray_value_without_dark_noise[mean_without_sat_indices]

    # Steigung und Y-Achsenabschnitt der Geraden
    slope, intercept, _, _, _ = stats.linregress(mean_without_sat, variance_without_sat)

    # Anfang und Ende der Geraden bestimmen
    line_begin = intercept
    line_end = intercept + slope * x_end

    plt.plot([x_begin, x_end], [line_begin, line_end])

    plt.show()

def main():
    #plot_mean_of_photons()
    plot_photo_transfer()

if __name__ == '__main__':
    main()
