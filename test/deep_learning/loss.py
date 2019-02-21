import tensorflow as tf
import numpy as np
from typing import List
from core.deep_learning.loss import CrossEntropy
from core.deep_learning.abstract_operator import AbstractLoss
import core.deep_learning.env as env


def test_predict(self : tf.test.TestCase, sess: tf.Session, loss : AbstractLoss, output_network : np.ndarray,
                 expected_output : np.ndarray):

    """Test the loss prediction tensor"""

    loss.output_network = tf.placeholder(tf.float32, output_network.shape)
    loss._set_predict()
    output = sess.run(loss.y_predict, {loss.output_network: output_network})
    self.assertAllEqual(output, expected_output)

def test_loss(self : tf.test.TestCase, sess : tf.Session, loss : AbstractLoss, output_network : np.ndarray,
              expected_loss : float, y : np.ndarray, rtol : float = 1e-6):

    """Test the loss computation"""

    loss.output_network = tf.placeholder(tf.float32, output_network.shape)
    loss.y = tf.placeholder(tf.float32, y.shape)
    loss._set_loss()

    output = sess.run(loss.loss, {loss.output_network: output_network, loss.y: y})
    self.assertAllClose(output, expected_loss, rtol=rtol)


def test_restore(self: tf.test.TestCase, loss: AbstractLoss, output_network_shape : List[int], y_shape : List[int],
                 tensors: List[str] = ("loss", "loss_opt", "y", "output_network", "y_predict"), weights : List = []):
    """This methods allow to test the restoration process of a Layer"""

    output_network = tf.placeholder(tf.float32, output_network_shape)
    y = tf.placeholder(tf.float32, y_shape)

    # create tensor inside the tensorflow graph
    loss.build(y, output_network, weights)

    # Set all restored attributes to None
    old_tensor = {}
    for var in tensors:
        old_tensor[var] = loss.__dict__[var]
        loss.__dict__[var] = None

    # Set the RESTORE environment variable to True
    env.RESTORE = True

    # Restore tensor
    loss.build(y, output_network, weights)

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
            expected = (3 * (1 ** 2) + 2 * (2 ** 2))/2
            self.assertEqual(loss.penality.eval(), expected)


class TestCrossEntropy(tf.test.TestCase):

    def testComputePredict(self):

        output_network = np.array([[1., 0.5],
                                   [2., 0.],
                                   [3.4, 10.]])
        loss = CrossEntropy()

        with self.test_session() as sess:
            expected_output = np.array([0, 0, 1])
            test_predict(self, sess, loss, output_network, expected_output)

    def testComputeLoss(self):

        output_network = np.array([[1., 0.5],
                                   [2., 0.],
                                   [3.4, 10.]])

        y = np.array([[1, 0],
                      [0, 1],
                      [1, 0]])

        loss = CrossEntropy()
        np_loss = np.exp(output_network)
        np_loss = np_loss / np_loss.sum(1).reshape(-1, 1)
        np_loss = - y * np.log(np_loss)
        expected_loss = np_loss.sum(1).mean()

        with self.test_session() as sess:
            test_loss(self, sess, loss, output_network, expected_loss, y, rtol=1e-6)

    def testRestore(self):

        loss = CrossEntropy()
        test_restore(self, loss, [100, 10], [100, 10], tensors=["loss", "loss_opt", "y", "output_network", "y_predict"])












