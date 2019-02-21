import tensorflow as tf
from typing import List

from core.deep_learning.abstract_operator import AbstractLoss


class CrossEntropy(AbstractLoss):

    """
    Build a cross entropy loss function for classification problems. Given an output network, the layyer compute
    prediction by taking the maximum score label as a prediction and learn weight by minimising the entropy function.

    This loss layer used only all loss attributes.
    """

    def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None,
                 name : str = "cross_entropy"):

        super().__init__(penalization_rate, penalization_type, name)

    def check_input(self):
        super().check_input()
        assert len(self.y.shape) == 2
        assert len(self.output_network.shape) == 2

    def _set_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(logits=self.output_network, onehot_labels=self.y)

    def _set_predict(self):
        self.y_predict = tf.argmax(self.output_network, 1)

    def build(self, y: tf.Tensor, output_network: tf.Tensor, weights : List[tf.Variable]):

        return super().build(y, output_network, weights)

    def restore(self):
        super().restore()













