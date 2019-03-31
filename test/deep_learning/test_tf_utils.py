import unittest

import tensorflow as tf

from core.deep_learning.tf_utils import get_act_funct

X = [
    [1., -2., 3.],
    [10., 20., -30.]
]


class TestActivationFunction(tf.test.TestCase):
    def testRelu(self):
        with self.test_session():
            expected = [
                [1, 0, 3],
                [10, 20, 0]
            ]

            self.assertAllEqual(get_act_funct('relu')(X).eval(), expected)

    def testSigmoid(self):
        with self.test_session():
            expected = [
                [0.731, 0.119, 0.953],
                [1., 1., 0.]
            ]

            self.assertAllClose(get_act_funct('sigmoid')(X).eval().round(3), expected, rtol=1e-3)

    def testTanh(self):
        with self.test_session():
            expected = [
                [0.762, -0.964, 0.995],
                [1., 1., -1.]
            ]

            self.assertAllClose(get_act_funct('tanh')(X).eval().round(3), expected, rtol=1e-3)

    def testError(self):
        self.assertRaises(TypeError, get_act_funct, X, 'unknow_name')


if __name__ == '__main__':
    unittest.main()
