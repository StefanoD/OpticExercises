from unittest import TestCase
from src import libcore
import numpy as np


class TestLibCore(TestCase):

    def test_get_time_variance(self):
        matrix1 = np.ones((5, 5))
        matrix2 = np.ones((5, 5)) * 2

        time_variance = libcore.get_time_variance(matrix1, matrix2)

        self.assertEqual(time_variance, 0.5)

    def test_mean(self):
        matrix1 = np.ones((5, 5))
        matrix2 = np.ones((5, 5)) * 2

        mean = libcore.get_mean([matrix1, matrix2])

        self.assertEqual(mean, 0.5)