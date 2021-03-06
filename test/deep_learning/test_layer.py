import unittest
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.layer import BaseLayer, FullyConnected, Conv1d, MinMax, Conv2d, Pool2d, Res2d

# 2d and 3d array for testing
X_2d = np.array([
    [1., -2., 3.],
    [10., 20., -30.]])

X_3d = np.array([
    [[1., -2., 3.],
     [10., 20., -30.]],

    [[0, 0, 0],
     [1, 2, -3]]])

X_4d = np.array([
    [[[1., -2., 3.], [10., 20., -30.]],
     [[0, 0, 0], [1, 2, -3]]],

    [[[1., -2., 3.], [10., 20., -30.]],
     [[0, 0, 0], [1, 2, -3]]]])


def test_operator(self: tf.test.TestCase, sess: tf.Session, layer: BaseLayer, x_input: np.ndarray,
                  expected_output: np.ndarray):
    """This method allows to test the operator of a layer """

    layer.x = tf.placeholder(tf.float32, x_input.shape)
    layer._operator()
    output = sess.run(layer.x_out, {layer.x: x_input})
    self.assertAllEqual(output, expected_output)


def test_restore(self: tf.test.TestCase, layer: BaseLayer, input_shape: List[int],
                 tensors: List[str] = ("x", "x_out"), build_args: Tuple = ()):
    """This methods allows to test the restoration process of a Layer"""

    x_input = tf.placeholder(tf.float32, input_shape)

    # create tensor inside the tensorflow graph
    layer.build(x_input, *build_args)

    # Set all restored attribute to None
    old_tensor = {}
    for var in tensors:
        old_tensor[var] = layer.__dict__[var]
        layer.__dict__[var] = None

    # Set the RESTORE environment variable to True
    env.RESTORE = True

    # Restore tensor
    layer.build(x_input, *build_args)

    env.RESTORE = False

    # assert the restored tensor has the same type, shape and name as before
    for var in tensors:
        self.assertTrue(old_tensor[var].dtype == layer.__dict__[var].dtype)
        self.assertTrue(old_tensor[var].shape == layer.__dict__[var].shape)
        self.assertTrue(old_tensor[var].name == layer.__dict__[var].name)


class TestFcLayer(tf.test.TestCase):

    def test_operator(self):
        with self.test_session() as sess:
            layer = FullyConnected(size=1)
            layer.w = [[1], [0.], [-1.]]
            layer.b = [1.]

            expected_output = np.array([[-1.], [41.]])
            test_operator(self, sess, layer, X_2d, expected_output)

    def test_restore(self):
        layer = FullyConnected(size=1)
        test_restore(self, layer, [100, 3], ["w", "b", "x", "x_out"])


class TestConv1dLayer(tf.test.TestCase):

    def test_operator(self):
        with self.test_session() as sess:
            w = [
                [[1., -1.],
                 [1., -1.],
                 [1., -1.]]]

            b = [1., -1.]

            # With bias
            layer = Conv1d(width=1, channels=2, stride=1, padding="VALID", add_bias=True)
            layer.w = w
            layer.b = b

            expected_output = np.array([
                [[3., -3.],
                 [1., -1.]],

                [[1., -1.],
                 [1., -1.]]])

            test_operator(self, sess, layer, X_3d, expected_output)

            # Without bias
            layer = Conv1d(width=1, channels=2, stride=1, padding="VALID", add_bias=False)
            layer.w = w
            layer.b = b

            expected_output = np.array([
                [[2., -2.],
                 [0., 0.]],

                [[0., 0.],
                 [0., 0.]]])

            test_operator(self, sess, layer, X_3d, expected_output)

    def test_restore(self):
        # With bias
        layer = Conv1d(width=1, channels=2, stride=1, padding="VALID", add_bias=True, name="with_bias")
        test_restore(self, layer, [100, 1, 3], ["w", "b", "x", "x_out"])
        # Without bias
        layer = Conv1d(width=1, channels=2, stride=1, padding="VALID", add_bias=False, name="without_bias")
        test_restore(self, layer, [100, 1, 3], ["w", "x", "x_out"])


class TestMinMaxLayer(tf.test.TestCase):

    def test_operator(self):
        with self.test_session() as sess:
            layer = MinMax(1)
            expected_output = np.array([
                [-2., 3.],
                [-30., 20.]])

            test_operator(self, sess, layer, X_2d, expected_output)

    def test_restore(self):
        layer = MinMax(1)
        test_restore(self, layer, [100, 3], ["x", "x_out"])


class TestConv2dLayer(tf.test.TestCase):

    def test_operator(self):
        # Simple Conv2d
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 3)).astype(np.float32)
            w = np.ones((2, 2, 3, 1)).astype(np.float32)
            b = np.ones(1).astype(np.float32)

            layer = Conv2d(width=3, height=3, filter=1, stride=(1, 1), dilation=None, padding="VALID",
                           add_bias=True, act_funct=None, name="TestOp")
            layer.w = w
            layer.b = b

            expected_output = np.ones((10, 31, 31, 1)) * 13

            test_operator(self, sess, layer, X, expected_output)

        # with dilation
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 3)).astype(np.float32)
            X[:, 0, 1, 0] = 2
            w = np.ones((2, 2, 3, 1)).astype(np.float32)
            b = np.ones(1).astype(np.float32)

            layer = Conv2d(width=3, height=3, filter=1, stride=(1, 2), dilation=(1, 1), padding="VALID",
                           add_bias=True, act_funct=None, name="TestOpDilation")
            layer.w = w
            layer.b = b

            expected_output = np.ones((10, 30, 15, 1)) * 13

            test_operator(self, sess, layer, X, expected_output)

    def test_restore(self):
        # With bias
        layer = Conv2d(width=1, height=1, filter=1, stride=(1, 1), dilation=None, padding="VALID",
                       add_bias=True, act_funct=None, name="TestRestore")
        test_restore(self, layer, [None, 16, 16, 3], ["w", "b", "x", "x_out"])


class TestPool2D(tf.test.TestCase):

    def test_operator_min_max(self):
        # Simple Pool2d MAX
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 1)).astype(np.float32)
            X[0, 0, 0, 0] = 10

            layer = Pool2d(width=3, height=3, stride=(1, 1), dilation=None, padding="VALID",
                           pooling_type="MAX", name="TestOp")

            expected_output = np.ones((10, 30, 30, 1))
            expected_output[0, 0, 0, 0] = 10

            test_operator(self, sess, layer, X, expected_output)

        # Simple Poolv2d MIN
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 1)).astype(np.float32)
            X[0, 0, 0, 0] = 10

            layer = Pool2d(width=3, height=3, stride=(1, 1), dilation=None, padding="VALID",
                           pooling_type="MIN", name="TestOp")

            expected_output = np.ones((10, 30, 30, 1))

            test_operator(self, sess, layer, X, expected_output)

        # with dilation
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 1)).astype(np.float32)
            X[:, 0, 1, 0] = 2

            layer = Pool2d(width=3, height=3, stride=(1, 2), dilation=(1, 1), padding="VALID",
                           pooling_type="MAX", name="TestDilation")

            expected_output = np.ones((10, 28, 14, 1))

            test_operator(self, sess, layer, X, expected_output)

    def test_operator_avg(self):
        # Simple Pool2d AVG
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 1)).astype(np.float32)

            layer = Pool2d(width=3, height=3, stride=(1, 1), dilation=None, padding="VALID",
                           pooling_type="AVG", name="TestOp")

            expected_output = np.ones((10, 30, 30, 1))

            test_operator(self, sess, layer, X, expected_output)

    def test_restore(self):
        layer = Pool2d(width=3, height=3, stride=(1, 1), dilation=None, padding="VALID",
                       pooling_type="MAX", name="TestRestore")
        test_restore(self, layer, [None, 16, 16, 3], ["x", "x_out"])


class TestRes2dLayer(tf.test.TestCase):

    def test_operator(self):
        # Simple Res2d
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 3)).astype(np.float32)
            X_lag = np.ones((10, 32, 32, 3)).astype(np.float32)

            layer = Res2d(name="TestIdentity")
            layer.x_lag = X_lag
            expected_output = np.ones((10, 32, 32, 3)) * 2

            test_operator(self, sess, layer, X, expected_output)

        # Reshape Res2d
        with self.test_session() as sess:
            X = np.ones((10, 30, 30, 3)).astype(np.float32)
            X_lag = np.ones((10, 32, 32, 3)).astype(np.float32)

            layer = Res2d(width=3, height=3, stride=(1, 1), padding="VALID", pooling_type="MAX", name="TestReshape")
            layer.x_lag = X_lag
            expected_output = np.ones((10, 30, 30, 3)) * 2

            test_operator(self, sess, layer, X, expected_output)

        # Pad Channel Res2d
        with self.test_session() as sess:
            X = np.ones((10, 32, 32, 10)).astype(np.float32)
            X_lag = np.ones((10, 32, 32, 3)).astype(np.float32)

            layer = Res2d(name="TestPad")
            layer.x_lag = X_lag
            expected_output = np.ones((10, 32, 32, 10)) * 2
            expected_output[..., :-3] = 1

            test_operator(self, sess, layer, X, expected_output)

    def test_restore(self):
        # With bias
        layer = Res2d(name="TestIdentity")

        X_lag = tf.placeholder(tf.float32, [None, 16, 16, 3])
        test_restore(self, layer, [None, 16, 16, 3], ["x", "x_out", "x_lag"], (X_lag,))


if __name__ == '__main__':
    unittest.main()
