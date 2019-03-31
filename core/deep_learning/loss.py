from typing import Sequence

import tensorflow as tf

from core.deep_learning.abstract_operator import AbstractLoss


class CrossEntropy(AbstractLoss):
    """
    Build a cross entropy loss function for classification problems. Given an output network, the layer compute
    prediction by taking the maximum score label as a prediction and learn weight by minimising the entropy function.

    This loss layer used only all loss attributes.
    """

    def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None,
                 name: str = "cross_entropy"):
        super().__init__(penalization_rate, penalization_type, name)

    def check_input(self):
        super().check_input()
        assert len(self.y.shape) == 2
        assert len(self.x_out.shape) == 2

    def _set_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(logits=self.x_out, onehot_labels=self.y)

    def _set_predict(self):
        self.y_pred = tf.argmax(self.x_out, 1)

    def build(self, y: tf.Tensor, x_out: tf.Tensor, weights: Sequence[tf.Variable]):
        return super().build(y, x_out, weights)

    def restore(self):
        super().restore()


class MeanSquareError(AbstractLoss):
    """
    Build a mean square error loss function allowing to learn one or many objective function in same time.

    """

    def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None,
                 name: str = "cross_entropy"):
        super().__init__(penalization_rate, penalization_type, name)

    def check_input(self):
        super().check_input()
        assert len(self.y.shape) == 2
        assert len(self.x_out.shape) == 2

    def _set_loss(self):
        self.loss = tf.pow(tf.subtract(self.y, self.x_out), 2)

        if self.y.shape[1] > 1:
            self.loss = tf.reduce_sum(self.loss, 1)

        self.loss = tf.reduce_mean(self.loss)

    def _set_predict(self):
        self.y_pred = self.x_out

    def build(self, y: tf.Tensor, x_out: tf.Tensor, weights: Sequence[tf.Variable]):
        return super().build(y, x_out, weights)

    def restore(self):
        super().restore()
