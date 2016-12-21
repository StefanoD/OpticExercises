import libcore
import matplotlib.pyplot as plt

"""Berechnete Bestrahlungsstärke E. Siehe Rechnung in Ordner MessungenAufgabe_1-2"""
IRRADIANCE_WATT_PER_SQUARE_METER = 0.121705797795879

"""Wellenlänge der verwendeten grünen LED"""
WAVELENGTH_METER = 0.000000525

"""Pixelgröße aus Datenblatt entnommen"""
PIXEL_AREA_METER = 0.0000045 ** 2

TIME_OF_EXPOSURE_SECONDS = [0.00002,
                            0.001645,
                            0.00327,
                            0.004895,
                            0.00652,
                            0.008145,
                            0.00977,
                            0.011395,
                            0.01302]


def plot_mean_of_photons():
    mean_of_photons_for_texp = []

    for texp in TIME_OF_EXPOSURE_SECONDS:
        mean_of_photons = libcore.get_mean_of_photons(PIXEL_AREA_METER,
                                        IRRADIANCE_WATT_PER_SQUARE_METER,
                                        texp,
                                        WAVELENGTH_METER)

        mean_of_photons_for_texp.append(mean_of_photons)

    plt.plot(TIME_OF_EXPOSURE_SECONDS, mean_of_photons_for_texp, 'ro')
    plt.xlabel('Time of Exposure [s]')
    plt.ylabel('Mean of Photons per Pixel')

    plt.show()


def main():
    plot_mean_of_photons()

if __name__ == '__main__':
    main()
