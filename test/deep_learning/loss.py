from typing import List

import numpy as np
import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.abstract_operator import AbstractLoss
from core.deep_learning.loss import CrossEntropy


def test_predict(self: tf.test.TestCase, sess: tf.Session, loss: AbstractLoss, x_out: np.ndarray,
                 expected_output: np.ndarray):
    """Test the loss prediction tensor"""

    loss.x_out = tf.placeholder(tf.float32, x_out.shape)
    loss._set_predict()
    output = sess.run(loss.y_pred, {loss.x_out: x_out})
    self.assertAllEqual(output, expected_output)


def test_loss(self: tf.test.TestCase, sess: tf.Session, loss: AbstractLoss, x_out: np.ndarray,
              expected_loss: float, y: np.ndarray, rtol: float = 1e-6):
    """Test the loss computation"""

    loss.x_out = tf.placeholder(tf.float32, x_out.shape)
    loss.y = tf.placeholder(tf.float32, y.shape)
    loss._set_loss()

    output = sess.run(loss.loss, {loss.x_out: x_out, loss.y: y})
    self.assertAllClose(output, expected_loss, rtol=rtol)


def test_restore(self: tf.test.TestCase, loss: AbstractLoss, x_out_shape: List[int], y_shape: List[int],
                 tensors: List[str] = ("loss", "loss_opt", "y", "x_out", "y_pred"), weights: List = []):
    """This methods allow to test the restoration process of a Layer"""

    x_out = tf.placeholder(tf.float32, x_out_shape)
    y = tf.placeholder(tf.float32, y_shape)

    # create tensor inside the tensorflow graph
    loss.build(y, x_out, weights)

    # Set all restored attributes to None
    old_tensor = {}
    for var in tensors:
        old_tensor[var] = loss.__dict__[var]
        loss.__dict__[var] = None

    # Set the RESTORE environment variable to True
    env.RESTORE = True

    # Restore tensor
    loss.build(y, x_out, weights)

    env.RESTORE = False

    # assert the restored tensor has the same type, shape and name as before
    for var in tensors:
        self.assertTrue(old_tensor[var].dtype == loss.__dict__[var].dtype)
        self.assertTrue(old_tensor[var].shape == loss.__dict__[var].shape)
        self.assertTrue(old_tensor[var].name == loss.__dict__[var].name)


class TesAbstractLoss(tf.test.TestCase):

    def testL2Penalization(self):
        W0 = np.array([1., 1., 1.])
        W1 = np.array([2., 2.])
        list_weight = [W0, W1]

        loss = CrossEntropy(penalization_type="L2")
        loss.weights = list_weight
        loss._compute_penalization()

        with self.test_session():
            expected = (3 * (1 ** 2) + 2 * (2 ** 2)) / 2
            self.assertEqual(loss.penality.eval(), expected)

    def testL1Penalization(self):
        W0 = np.array([1., 1., -1.])
        W1 = np.array([-2., -2.])
        list_weight = [W0, W1]

        loss = CrossEntropy(penalization_type="L1")
        loss.weights = list_weight
        loss._compute_penalization()
        with self.test_session():
            expected = (1 + 1 + 1) + (2 + 2)
            self.assertEqual(loss.penality.eval(), expected)


class TestCrossEntropy(tf.test.TestCase):

    def testComputePredict(self):
        x_out = np.array([[1., 0.5],
                          [2., 0.],
                          [3.4, 10.]])
        loss = CrossEntropy()

        with self.test_session() as sess:
            expected_output = np.array([0, 0, 1])
            test_predict(self, sess, loss, x_out, expected_output)

    def testComputeLoss(self):
        x_out = np.array([[1., 0.5],
                          [2., 0.],
                          [3.4, 10.]])

        y = np.array([[1, 0],
                      [0, 1],
                      [1, 0]])

        loss = CrossEntropy()
        np_loss = np.exp(x_out)
        np_loss = np_loss / np_loss.sum(1).reshape(-1, 1)
        np_loss = - y * np.log(np_loss)
        expected_loss = np_loss.sum(1).mean()

        with self.test_session() as sess:
            test_loss(self, sess, loss, x_out, expected_loss, y, rtol=1e-6)

    def testRestore(self):
        loss = CrossEntropy()
        test_restore(self, loss, [100, 10], [100, 10], tensors=["loss", "loss_opt", "y", "x_out", "y_pred"])
