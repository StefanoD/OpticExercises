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

QUANTIZATION_NOISE = 1.0/12.0


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


def get_mean_dark_grey_value():
    # ../MessungenAufgabe_1-2/geschlossen/*
    path_closed = join(join(join("..", "MessungenAufgabe_1-2"), "geschlossen"), "*")

    images_closed = libcore.get_sorted_images(path_closed)

    mean_closed = libcore.get_mean_of_two_images(images_closed)
    return np.array(mean_closed)


def get_mean_grey_values():
    # ../MessungenAufgabe_1-2/offen/*
    path_open = join(join(join("..", "MessungenAufgabe_1-2"), "offen", "*"))

    images_open = libcore.get_sorted_images(path_open)

    mean_open = libcore.get_mean_of_two_images(images_open)
    return np.array(mean_open)


def get_variance_gray_values():
    # ../MessungenAufgabe_1-2/offen/*
    path_open = join(join(join("..", "MessungenAufgabe_1-2"), "offen", "*"))

    # Verarbeite images_open
    images_open = libcore.get_sorted_images(path_open)

    variance_open = libcore.get_time_variance_of_two_images(images_open)
    return np.array(variance_open)


def get_variance_of_dark_gray_values():
    # ../MessungenAufgabe_1-2/geschlossen/*
    path_closed = join(join(join("..", "MessungenAufgabe_1-2"), "geschlossen"), "*")

    images_closed = libcore.get_sorted_images(path_closed)

    variance_closed = libcore.get_time_variance_of_two_images(images_closed)
    return np.array(variance_closed)


def plot_photon_transfer():
    # Hole Mittelwert und Varianz, die aufsteigend nach der Belichtungszeigt sortiert sind
    # Varianz
    variance_gray_value = get_variance_gray_values()
    variance_dark_gray_value = get_variance_of_dark_gray_values()

    variance_gray_value_without_dark_noise = variance_gray_value - variance_dark_gray_value

    # Mittelwert
    mean_grey_values = get_mean_grey_values()
    mean_dark_grey_values = get_mean_dark_grey_value()

    mean_gray_value_without_dark_noise = mean_grey_values - mean_dark_grey_values

    # Limits der Axen setzen
    x_begin = 0
    x_end = 300
    plt.xlim(x_begin, x_end)
    plt.ylim(-1, np.max(variance_gray_value_without_dark_noise) * 1.3)

    # Plotte mittlere Grauwerte und Varianz ohne Dunkelstrom
    plt.plot(mean_gray_value_without_dark_noise, variance_gray_value_without_dark_noise, 'ro')
    plt.xlabel('gray value - dark value $\mu_{{y}} - \mu_{{y.dark}}$ (DN)')
    plt.ylabel('variance gray value $\sigma^2_{{y}} - \sigma^2_{{y.dark}}$ (DN²)')

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
    # Die Steigung ist gleichzeitig auch der Gain K.
    slope, intercept, _, _, stderr = stats.linregress(mean_without_sat, variance_without_sat)
    system_gain = slope

    # Anfang und Ende der Geraden bestimmen
    line_begin = intercept
    line_end = intercept + slope * mean_sat_begin

    # Zeichne durchgezogene Linie bis Sättigungsbeginn
    plt.plot([x_begin, mean_sat_begin], [line_begin, line_end])

    # Zeichne gestrichelte Linie ab Sättigungsbeginn
    plt.plot([mean_sat_begin, 255], [line_end, intercept + slope * 255], 'b--')

    dark_signal = libcore.get_dark_signal(variance_dark_gray_value[0],
                                          QUANTIZATION_NOISE, system_gain)

    # Dunkel-Signal und System Gain in Plot einfügen
    stderr_percent = stderr / system_gain * 100
    plt.text(65, -0.7, r'$\sigma^2_{{y.dark}} = {:.2f} DN^2, K = {:.4} \pm {:.2}\%$'.format(dark_signal,
                                                                                            system_gain,
                                                                                            stderr_percent))

    plt.title("Photon transfer")

    plt.show()
    plt.clf()

    return system_gain, saturation_index, dark_signal


def get_mean_of_photons_per_pixel_and_exposure_time():
    # Photonen pro Pixel und Belichtungszeit berechnen
    mean_of_photons_for_texp = []

    for texp in TIME_OF_EXPOSURE_MS:
        texp_sec = texp / 1000
        mean_of_photons = libcore.get_mean_of_photons(PIXEL_AREA_METER,
                                                      IRRADIANCE_WATT_PER_SQUARE_METER,
                                                      texp_sec,
                                                      WAVELENGTH_METER)

        mean_of_photons_for_texp.append(mean_of_photons)

    return np.array(mean_of_photons_for_texp)


def plot_sensivity(system_gain, saturation_index):
    # Mittelwert
    mean_grey_values = get_mean_grey_values()
    mean_dark_grey_values = get_mean_dark_grey_value()

    mean_gray_value_without_dark_noise = mean_grey_values - mean_dark_grey_values

    mean_of_photons_for_texp = get_mean_of_photons_per_pixel_and_exposure_time()

    x_end = np.max(mean_of_photons_for_texp) * 1.05

    # Plotte mittlere Grauwerte und Varianz ohne Dunkelstrom
    plt.plot(mean_of_photons_for_texp, mean_gray_value_without_dark_noise, 'ro')
    plt.xlabel('irradiation (photons/pixel)')
    plt.ylabel('gray value - dark value $\mu_{{y}} - \mu_{{y.dark}}$ (DN)')

    # Lineare Regeression der steigenden Varianz bilden.
    # Die Sättigung soll dabei nicht miteinfließen.
    # Man geht davon aus, dass bei 70% des Sättigungspunktes die Sättigung anfängt
    irradiation_sat_begin = mean_of_photons_for_texp[saturation_index] * 0.7

    # Ermittle die Indizes der Matrix-Einträge die einen kleineren Wert als mean_sat_begin haben
    without_sat_indices = mean_of_photons_for_texp < irradiation_sat_begin

    # Mean-Matrix von der Sättigung bereinigt
    mean_without_sat = mean_gray_value_without_dark_noise[without_sat_indices]
    irradiation_without_sat = mean_of_photons_for_texp[without_sat_indices]

    # Steigung und Y-Achsenabschnitt der Geraden
    # Die Steigung ist gleichzeitig auch der Gain K.
    slope, intercept, _, _, stderr = stats.linregress(irradiation_without_sat, mean_without_sat)
    responsivity = slope

    # Anfang und Ende der Geraden bestimmen
    line_begin = intercept
    line_end = intercept + slope * irradiation_sat_begin

    # Zeichne durchgezogene Linie bis Sättigungsbeginn
    plt.plot([0, irradiation_sat_begin], [line_begin, line_end])

    # Zeichne gestrichelte Linie ab Sättigungsbeginn
    plt.plot([irradiation_sat_begin, x_end], [line_end, intercept + slope * x_end], 'b--')

    # Dunkel-Bild bei kleinstmöglichester Belichtungszeit hinzufügen
    quantum_efficiency = responsivity / system_gain
    plt.text(37000, 10, r'$\mu_{{y.dark}} = {:.2f} DN, \eta = {:.2f}$'.format(mean_dark_grey_values[0],
                                                                              quantum_efficiency))

    plt.title("Sensitivity")

    plt.show()
    plt.clf()

    return quantum_efficiency


def plot_SNR(system_gain, quantum_efficiency, variance_dark_signal):
    # Mittelwert
    mean_grey_values = get_mean_grey_values()
    mean_dark_grey_values = get_mean_dark_grey_value()

    mean_gray_value_without_dark_noise = mean_grey_values - mean_dark_grey_values

    variance_gray_value = get_variance_gray_values()

    snr_matrix = mean_gray_value_without_dark_noise / variance_gray_value
    snr_ideal_matrix = np.sqrt(mean_gray_value_without_dark_noise)

    mean_of_photons_for_texp = get_mean_of_photons_per_pixel_and_exposure_time()

    # Plotte mittlere Grauwerte und Varianz ohne Dunkelstrom
    plt.loglog(mean_of_photons_for_texp, snr_matrix, 'ro', label="Measurements")
    plt.xlabel('irradiation (photons/pixel)')
    plt.ylabel('SNR')

    plt.loglog(mean_of_photons_for_texp, snr_ideal_matrix, label="Ideal SNR")

    # Sättingspunkt ist an der Stelle der maximalen Varianz
    saturation_index = np.argmax(snr_matrix)

    # Lineare Regeression der steigenden Varianz bilden.
    # Die Sättigung soll dabei nicht miteinfließen.
    # Man geht davon aus, dass bei 70% des Sättigungspunktes die Sättigung anfängt
    irradiation_sat_begin = mean_of_photons_for_texp[saturation_index] * 0.7

    # Ermittle die Indizes der Matrix-Einträge die einen kleineren Wert als mean_sat_begin haben
    without_sat_indices = mean_of_photons_for_texp < irradiation_sat_begin

    # Matrizen von der Sättigung bereinigt
    snr_without_sat = snr_matrix[without_sat_indices]
    irradiation_without_sat = mean_of_photons_for_texp[without_sat_indices]

    # Interpolation der gemessenen Punkte.
    from scipy.interpolate import interp1d

    new_x = np.insert(irradiation_without_sat, 0, 0)
    new_y = np.insert(snr_without_sat, 0, 0)

    f2 = interp1d(new_x, new_y, kind='cubic')

    new_x2 = np.insert(irradiation_without_sat, 0, 1)
    plt.loglog(new_x2, f2(new_x2), '--', label="fit")

    # Berechne minimale Bestrahlungsstärke mit SNR = 1
    variance_dark_gray_value = get_variance_of_dark_gray_values()
    minimum_irradiation = libcore.get_minimum_irradiation(quantum_efficiency, variance_dark_gray_value[0], system_gain)

    plt.loglog([minimum_irradiation, minimum_irradiation], plt.ylim(),
               color='red', linewidth=1.5, linestyle="--")

    # Berechne theoretische SNR-Kurve
    # Skript 2, S. 18
    snr_theory_values = mean_of_photons_for_texp * quantum_efficiency / np.sqrt(variance_dark_signal + (QUANTIZATION_NOISE / (system_gain ** 2)) +
                              quantum_efficiency * mean_of_photons_for_texp)

    plt.loglog(mean_of_photons_for_texp, snr_theory_values, label="theor. limit")
    # Berechne über lineare Regression den Schwellwert für SNR = 1

    slope, intercept, _, _, _ = stats.linregress(mean_of_photons_for_texp, snr_theory_values)

    plt.title("SNR")

    plt.legend(loc='upper left')

    plt.show()

def main():
    #plot_mean_of_photons()
    system_gain, saturation_index, variance_dark_signal = plot_photon_transfer()
    quantum_efficiency = plot_sensivity(system_gain, saturation_index)
    plot_SNR(system_gain, quantum_efficiency, variance_dark_signal)

if __name__ == '__main__':
    main()
