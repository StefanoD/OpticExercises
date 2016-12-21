import libcore
import matplotlib.pyplot as plt

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


def main():
    plot_mean_of_photons()

if __name__ == '__main__':
    main()
