import unittest

import numpy as np

from core.deep_learning.image_processing import resenet_50


class testResNet50(unittest.TestCase):

    def testPredictOutputConv(self):
        img = np.random.uniform(0, 1, (2, 224, 224, 3))

        output = resenet_50(img)

        self.assertEqual(output.shape, (2, 7, 7, 2048))
