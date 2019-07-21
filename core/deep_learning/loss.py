from typing import Sequence, Union, Tuple, Optional

import tensorflow as tf

from core.deep_learning.base_operator import BaseLoss, BaseOperator
from core.deep_learning.tf_utils import get_tf_tensor


class CrossEntropy(BaseLoss):
    """
    Build a cross entropy loss function for classification problems. Given an output network which predict :math:`C`
    class, learns network parameters by minimising for all observations the following function function:

    .. math::

        - \\sum_{c=1}^{C} y_{c} \\log(p_{c}) \\quad \\text{with} \\quad p_{c} = \\frac{\\exp(o_{c})}{\\sum_{i=1}^{C} \\exp(o_i)}

    The final predicted label is the output network :math:`o_{c}` with the highest value.

    Args
    ----

        penalization_rate: tf.Tensor, float
            Penalization rate to apply to regularization terms.

        penalization_type: str, None
            Name of the penalization to use on network weight. If None no regularization is used.

        name: str
            Layer name for the Tensorflow graph.


    """

    def __init__(self, penalization_rate: Union[tf.Tensor, float] = 0.5, penalization_type: Optional[str] = None,
                 name: str = "cross_entropy"):
        super().__init__(penalization_rate, penalization_type, name)

    def check_input(self) -> None:
        """
        Check the shape of all input tensor.
        """

        super().check_input()
        assert len(self.y.shape) == 2
        assert len(self.x_out.shape) == 2

    def _set_loss(self) -> None:
        """
        Set the loss tensor.
        """

        self.loss = tf.losses.softmax_cross_entropy(logits=self.x_out, onehot_labels=self.y)

    def _set_predict(self) -> None:
        """
        Set the prediction tensor equal to the argmax of the output network matrix.
        """

        self.y_pred = tf.argmax(self.x_out, 1)

    def build(self, y: tf.Tensor, x_out: tf.Tensor, weights: Sequence[tf.Variable]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build the cross entropy loss tensor and return the

        Args
        ----

            y: tf.Tensor
                Tensor of true label to predict.

            x_out: tf.Tensor
                Output network tensor which must have the save dimension as y.

            weights: Sequence[tf.Variable]
                Weights which must be regularized.

        Returns
        -------

            tf.Tensor
                Loss tensor to optimize.

            tf.Tensor
                Loss tensor without any regularization terms.

        """

        return super().build(y, x_out, weights)

    def restore(self) -> None:
        """
        Restore all loss tensor from the current graph.

        """
        super().restore()


class MeanSquareError(BaseLoss):
    """
    Build a mean square error loss function allowing to learn one or many objective function in same time. If the
    problem has :math:`C` objective variable to predict, the following function is minimize for each observation:

    .. math::

            \\frac{1}{C} \\sum_{c=1}^{C} (y_c - \\hat{y_c})^2


    Args
    ----

        penalization_rate: tf.Tensor, float
            Penalization rate to apply to regularization terms.

        penalization_type: str, None
            Name of the penalization to use on network weight. If None no regularization is used.

        name: str
            Layer name for the Tensorflow graph.
    """

    def __init__(self, penalization_rate: Union[tf.Tensor, float] = 0.5, penalization_type: Optional[str] = None,
                 name: str = "mean_square_error"):
        super().__init__(penalization_rate, penalization_type, name)

    def check_input(self) -> None:
        """
        Check all input tensors are 2 dimensional.
        """

        super().check_input()
        assert len(self.y.shape) == 2
        assert len(self.x_out.shape) == 2

    def _set_loss(self) -> None:
        """
        Set the loss tensor.
        """

        self.loss = tf.pow(tf.subtract(self.y, self.x_out), 2)

        if self.y.shape[1] > 1:
            self.loss = tf.reduce_sum(self.loss, 1)

        self.loss = tf.reduce_mean(self.loss)

    def _set_predict(self) -> None:
        """
        Set the prediction tensor equal to the output network.
        """

        self.y_pred = self.x_out

    def build(self, y: tf.Tensor, x_out: tf.Tensor, weights: Sequence[tf.Variable]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build the cross entropy loss tensor and return the

        Args
        ----

            y: tf.Tensor
                Tensor of true label to predict.

            x_out: tf.Tensor
                Output network tensor which must have the save dimension as y.

            weights: Sequence[tf.Variable]
                Weights which must be regularized.

        Returns
        -------

            tf.Tensor
                Loss tensor to optimize.

            tf.Tensor
                Loss tensor without any regularization terms.

        """
        return super().build(y, x_out, weights)

    def restore(self) -> None:
        """
        Restore all loss tensor from the current graph.
        """

        super().restore()


class GanLoss(BaseOperator):

    def __init__(self, name: str = "mean_square_error"):
        super().__init__(name)

        self.dis_out: Optional[tf.Tensor] = None
        self.dis_gen_out: Optional[tf.Tensor] = None

        self.dis_loss: Optional[tf.Tensor] = None
        self.gen_loss: Optional[tf.Tensor] = None

    def check_input(self) -> None:
        pass

    def _set_loss(self) -> None:
        """
        Set the loss tensor.
        """

        self.dis_out = tf.identity(self.dis_out, name="dis_out")
        self.dis_gen_out = tf.identity(self.dis_gen_out, name="dis_gen_out")

        self.dis_loss = -tf.reduce_mean(tf.log(self.dis_out + 1e-8) + tf.log(1. - self.dis_gen_out + 1e-8))
        self.gen_loss = -tf.reduce_mean(tf.log(self.dis_gen_out + 1e-8))

        self.dis_loss = tf.identity(self.dis_loss, name='dis_loss')
        self.gen_loss = tf.identity(self.gen_loss, name='gen_loss')

    def build(self, dis_out: tf.Tensor, dis_gen_out: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        super().build(dis_out, dis_gen_out)

        return self.dis_loss, self.gen_loss

    def _build(self, dis_out: tf.Tensor, dis_gen_out: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        self.dis_out = dis_out
        self.dis_gen_out = dis_gen_out

        self.check_input()

        self._set_loss()

        return self.dis_loss, self.gen_loss

    def restore(self) -> None:
        """
        Restore all loss tensor from the current graph.
        """

        self.dis_out = get_tf_tensor(name='dis_out')
        self.dis_gen_out = get_tf_tensor(name='dis_gen_out')
        self.dis_loss = get_tf_tensor(name='dis_loss')
        self.gen_loss = get_tf_tensor(name='gen_loss')
