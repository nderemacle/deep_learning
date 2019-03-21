from typing import List

import numpy as np
import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.layer import AbstractLayer, FcLayer, Conv1dLayer, MinMaxLayer

# 2d and 3d array for testing
X_2d = np.array([
    [1., -2., 3.],
    [10., 20., -30.]])

X_3d = np.array([
    [[1., -2., 3.],
     [10., 20., -30.]],

    [[0, 0, 0],
     [1, 2, -3]]])


def test_operator(self: tf.test.TestCase, sess: tf.Session, layer: AbstractLayer, x_input: np.ndarray,
                  expected_output: np.ndarray):
    """This method allow to test the operator of a layer """

    layer.x = tf.placeholder(tf.float32, x_input.shape)
    layer._operator()
    output = sess.run(layer.x_out, {layer.x: x_input})
    self.assertAllEqual(output, expected_output)


def test_restore(self: tf.test.TestCase, layer: AbstractLayer, input_shape: List[int],
                 tensors: List[str] = ("x", "x_out")):
    """This methods allow to test the restoration process of a Layer"""

    x_input = tf.placeholder(tf.float32, input_shape)

    # create tensor inside the tensorflow graph
    layer.build(x_input)

    # Set all restored attribute to None
    old_tensor = {}
    for var in tensors:
        old_tensor[var] = layer.__dict__[var]
        layer.__dict__[var] = None

    # Set the RESTORE environment variable to True
    env.RESTORE = True

    # Restore tensor
    layer.build(x_input)

    env.RESTORE = False

    # assert the restored tensor has the same type, shape and name as before
    for var in tensors:
        self.assertTrue(old_tensor[var].dtype == layer.__dict__[var].dtype)
        self.assertTrue(old_tensor[var].shape == layer.__dict__[var].shape)
        self.assertTrue(old_tensor[var].name == layer.__dict__[var].name)


class TestFcLayer(tf.test.TestCase):

    def test_operator(self):
        with self.test_session() as sess:
            layer = FcLayer(size=1)
            layer.w = [[1], [0.], [-1.]]
            layer.b = [1.]

            expected_output = np.array([[-1.], [41.]])
            test_operator(self, sess, layer, X_2d, expected_output)

    def test_restore(self):
        layer = FcLayer(size=1)
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
            layer = Conv1dLayer(filter_width=1, n_filters=2, stride=1, padding="VALID", add_bias=True)
            layer.w = w
            layer.b = b

            expected_output = np.array([
                [[3., -3.],
                 [1., -1.]],

                [[1., -1.],
                 [1., -1.]]])

            test_operator(self, sess, layer, X_3d, expected_output)

            # Without bias
            layer = Conv1dLayer(filter_width=1, n_filters=2, stride=1, padding="VALID", add_bias=False)
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
        layer = Conv1dLayer(filter_width=1, n_filters=2, stride=1, padding="VALID", add_bias=True, name="with_bias")
        test_restore(self, layer, [100, 1, 3], ["w", "b", "x", "x_out"])
        # Without bias
        layer = Conv1dLayer(filter_width=1, n_filters=2, stride=1, padding="VALID", add_bias=False, name="without_bias")
        test_restore(self, layer, [100, 1, 3], ["w", "x", "x_out"])


class TestMinMaxLayer(tf.test.TestCase):

    def test_operator(self):
        with self.test_session() as sess:
            layer = MinMaxLayer(1)
            expected_output = np.array([
                [-2., 3.],
                [-30., 20.]])

            test_operator(self, sess, layer, X_2d, expected_output)

    def test_restore(self):
        layer = MinMaxLayer(1)
        test_restore(self, layer, [100, 3], ["x", "x_out"])
