import libcore
import matplotlib.pyplot as plt
import numpy as np

from os.path import join
from scipy import stats
from scipy.ndimage import uniform_filter
from scipy.stats import norm
from matplotlib.patches import Circle

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
    plt.text(65, -0.7, r'$\sigma^2_{{y.dark}} = {:.2f} DN^2, K = {:.4} \pm {:.2}\%$'.format(variance_dark_gray_value[0],
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

    irradiation_sat = mean_of_photons_for_texp[saturation_index]
    dynamic_range = 20 * np.log10(irradiation_sat / minimum_irradiation)

    print("Dynamic Range: {:.2f} dB".format(dynamic_range))

    plt.show()


def get_power_spectrum(fft, axis):
    if axis == 0:
        other_axis = 1
    elif axis == 1:
        other_axis = 0

    return np.sqrt((1 / fft.shape[other_axis]) * np.sum((fft * np.conj(fft)), axis=other_axis))


def aufgabe3_4(system_gain):
    "MessungenAufgabe3/geschlossen"
    # ../MessungenAufgabe_3/geschlossen/*
    path_closed = join(join(join("..", "MessungenAufgabe3"), "geschlossen"), "*")
    closed_images = np.array(libcore.get_sorted_images(path_closed))
    dark_image = closed_images.mean(axis=0)
    dark_image_mean = dark_image.mean()
    dark_image_variance = np.var(dark_image)

    dark_temporal_variance_matrix = np.sum((closed_images - dark_image) ** 2, axis=0) / (len(dark_image) - 1)
    dark_temporal_variance_stack = dark_temporal_variance_matrix.mean()
    dark_temporal_std_stack = np.sqrt(dark_temporal_variance_stack)

    corrected_dark_spatial_variance = dark_image_variance - (dark_temporal_variance_stack / len(path_closed))
    corrected_dark_spatial_std = np.sqrt(corrected_dark_spatial_variance)

    DSNU = corrected_dark_spatial_std / system_gain

    print("DSNU: {:.2f} electrons".format(DSNU))

    # ../MessungenAufgabe_3/offen/*
    path_open = join(join(join("..", "MessungenAufgabe3"), "offen", "*"))
    open_images = np.array(libcore.get_sorted_images(path_open))
    white_image = open_images.mean(axis=0)
    white_image_mean = white_image.mean()
    white_image_variance = np.var(white_image)

    white_temporal_variance_matrix = np.sum((open_images - white_image) ** 2, axis=0) / (len(white_image) - 1)
    white_temporal_variance_stack = white_temporal_variance_matrix.mean()
    white_temporal_std_stack = np.sqrt(white_temporal_variance_stack)

    corrected_white_spatial_variance = white_image_variance - (white_temporal_variance_stack / len(open_images))

    PRNU = np.sqrt(corrected_white_spatial_variance - corrected_dark_spatial_variance) / (white_image_mean - dark_image_mean)
    print("PRNU: {:.2f}%".format(PRNU))

    """Spektrogramme berechnen"""
    white_image_diff = white_image - white_image_mean

    """Weiß Horizontal-PRNU"""
    axis = 1
    height, width = white_image_diff.shape
    fft_1D = np.fft.fft(white_image_diff, axis=axis) / np.sqrt(np.size(white_image_diff, axis=axis))
    fft_mean_1D = get_power_spectrum(fft_1D, axis=axis)
    plt.figure(1)
    plt.title("Horizontal Spectrogram PRNU")
    plt.legend(loc='upper left')

    plt.semilogy(fft_mean_1D[:len(fft_mean_1D) // 2], label="spectrogram")
    plt.semilogy([0, len(fft_mean_1D) // 2], [white_temporal_std_stack, white_temporal_std_stack], "g--", label="temp. std")
    plt.semilogy([0, len(fft_mean_1D) // 2], [PRNU, PRNU], "r--", label="spat. std")

    plt.xlabel('frequency')
    plt.ylabel('standard deviation (%)')

    """Weiß Vertikal-PRNU"""
    axis = 0
    fft_1D = np.fft.fft(white_image_diff, axis=axis) / np.sqrt(np.size(white_image_diff, axis=axis))
    fft_mean_1D = get_power_spectrum(fft_1D, axis=axis)
    plt.figure(2)
    plt.title("Vertical Spectrogram PRNU")
    plt.legend(loc='upper left')

    plt.semilogy(fft_mean_1D[:len(fft_mean_1D) // 2], label="spectrogram")
    plt.semilogy([0, len(fft_mean_1D) // 2], [white_temporal_std_stack, white_temporal_std_stack], "g--", label="temp. std")
    plt.semilogy([0, len(fft_mean_1D) // 2], [PRNU, PRNU], "r--", label="spat. std")

    plt.xlabel('frequency')
    plt.ylabel('standard deviation (%)')

    """Spektrogramm DSNU Horizontal"""
    axis = 1
    dark_image_diff = dark_image - dark_image_mean
    fft_1D = np.fft.fft(dark_image_diff, axis=axis) / np.sqrt(np.size(white_image_diff, axis=axis))
    fft_mean_1D = get_power_spectrum(fft_1D, axis=axis)
    plt.figure(3)
    plt.title("Horizontal Spectrogram DSNU")
    plt.yscale("log")
    plt.plot(fft_mean_1D[:len(fft_mean_1D) // 2])
    plt.plot([0, len(fft_mean_1D) // 2], [dark_temporal_std_stack, dark_temporal_std_stack], "g--")
    plt.plot([0, len(fft_mean_1D) // 2], [corrected_dark_spatial_std, corrected_dark_spatial_std], "r--")

    plt.xlabel('frequency')
    plt.ylabel('standard deviation (DN)')

    """Spektrogramm DSNU Vertikal"""
    axis = 0
    dark_image_diff = dark_image - dark_image_mean
    fft_1D = np.fft.fft(dark_image_diff, axis=axis) / np.sqrt(np.size(white_image_diff, axis=axis))
    fft_mean_1D = get_power_spectrum(fft_1D, axis=axis)
    plt.figure(4)
    plt.title("Vertical Spectrogram DSNU")
    plt.yscale("log")
    plt.plot(fft_mean_1D[:len(fft_mean_1D) // 2])
    plt.plot([0, len(fft_mean_1D) // 2], [dark_temporal_std_stack, dark_temporal_std_stack], "g--")
    plt.plot([0, len(fft_mean_1D) // 2], [corrected_dark_spatial_std, corrected_dark_spatial_std], "r--")

    plt.xlabel('frequency')
    plt.ylabel('standard deviation (DN)')

    # Aufgabe 4
    white_dark_image_diff = white_image - dark_image
    lowpass_image = uniform_filter(white_dark_image_diff, 5)
    highpass_image = white_dark_image_diff - lowpass_image

    highpass_flattend = highpass_image.flatten()

    plt.figure(5)
    plt.title("Logarithmic histogram PRNU")
    plt.yscale("log")
    plt.hist(highpass_flattend, bins=255)

    variance_highpass = np.var(highpass_flattend)
    total_variance = white_temporal_variance_stack + variance_highpass
    mean_highpass = np.mean(highpass_flattend)

    x_left = norm.ppf(0.000000001, mean_highpass, total_variance)
    x_right = norm.ppf(0.999999999, mean_highpass, total_variance)
    x = np.linspace(x_left, x_right, 200)
    max_val = np.max(np.histogram(highpass_flattend)[0])

    """ Schwellwert definieren """
    threshold = norm.ppf(0.0000000000001, mean_highpass, total_variance)

    print("mean highpass: {}".format(mean_highpass))
    print("threshold: {}".format(threshold))

    fit_data = norm.pdf(x, mean_highpass, total_variance)
    max_val_fit = np.max(fit_data)
    fit_data *= (max_val / max_val_fit)

    plt.plot(x, fit_data, 'r-', lw=5, alpha=0.6, label='norm pdf')

    """Dead Pixel ermitteln"""

    dead_pixel_positions_low = np.where(highpass_image < threshold)
    dead_pixel_positions_high = np.where(highpass_image > abs(threshold))

    fig = plt.figure(6)
    plt.title("Dead Pixel")
    ax = fig.gca()
    plt.imshow(highpass_image, cmap=plt.get_cmap("Greys"))

    for y, x in zip(*dead_pixel_positions_low):
        ax.add_patch(Circle((x, y), 5))

    for y, x in zip(*dead_pixel_positions_high):
        ax.add_patch(Circle((x, y), 5))


    """DSNU"""
    plt.figure(7)
    plt.title("Logarithmic DSNU histogram")
    plt.yscale("log")
    plt.hist(dark_image.flatten(), bins=256)

    """Hot Pixel ermitteln"""

    threshold = 7

    fig = plt.figure(8)
    ax = fig.gca()
    plt.title("Hot Pixels")
    hot_pixel_positions = np.where(dark_image > threshold)

    plt.imshow(dark_image, cmap=plt.get_cmap("Greys"))

    for y, x in zip(*hot_pixel_positions):
        ax.add_patch(Circle((x, y), 5))

    plt.show()


def main():
    #plot_mean_of_photons()
    system_gain, saturation_index, variance_dark_signal = plot_photon_transfer()
    quantum_efficiency = plot_sensivity(system_gain, saturation_index)
    plot_SNR(system_gain, quantum_efficiency, variance_dark_signal)
    aufgabe3_4(system_gain)

if __name__ == '__main__':
    main()
