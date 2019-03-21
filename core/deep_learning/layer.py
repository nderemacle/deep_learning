from typing import Union

import numpy as np
import tensorflow as tf

from core.deep_learning.abstract_operator import AbstractLayer
from core.deep_learning.tf_utils import variable, get_tf_tensor
from core.utils.validation import check_tensor


class FcLayer(AbstractLayer):

    def __init__(self,
                 size: int,
                 act_funct: Union[str, None] = "relu",
                 keep_proba: Union[tf.Tensor, float] = 1.,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 is_training: Union[tf.Tensor, None] = None,
                 name: str = "fc",
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.001,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):
        """
        Deployed a fully connected layer having a size equal to it number of neurons. It inherit to all trainning
        methods implement in the AbstractLayer class and can used it.

        Args
        ----

            size : int,
                number of neurons to output

            act_funct : str
                name of the activation function to use. If None no activation function are used.

            keep_proba : float
                probability to keep a neuron activate during trainning if we want to apply the dropout method.

            batch_norm: bool
                If True apply the batch normalization after the _operator method.

            batch_renorm: bool
                Whether to used batch renormalization or not.

            is_training : Tensor
                Tensor indicating if data are used for training or for prediction. Useful for batch normalization.

            name : str
                name of the layer

            law_name : str
                name of the lax to use for weights and biases initialization

            law_param : float
                parameter of the initialization law

            decay: float
                Decay used to update the moving average of the batch norm. The moving average is used to learn the
                empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.)

            epsilon: float
                 Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

            decay_renorm: float
                Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

            rmin: Tensor or float
                Minimum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            rmax: Tensor or float
                Maximum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            dmax: Tensor or float
                When batch renormalization is used the scaled mu differences is clipped between (-dmax, dmax)

    Attributes
    ----------

            w : Variable with size (input_dim, size)
                Weights of the layer. Must be learnt.

            b : Variable with size (size)
                Biases of the layer. Must be learnt.
        """

        super().__init__(act_funct, keep_proba, batch_norm, batch_renorm, is_training, law_name, law_param, decay,
                         epsilon, decay_renorm, rmin, rmax, dmax, name)

        self.size = size
        self.w: tf.Variable = None
        self.b: tf.Variable = None

    def _check_input(self) -> None:
        """Assert the input tensor have shape 2"""

        check_tensor(self.x, shape_dim=2)

    def _init_variable(self, w_init: np.ndarray = None, b_init: np.ndarray = None) -> None:
        """Initialize the weight and biase. If init matrix are input variable are initialize using them.

        Attributes:

            w_int : np.array, None
                Matrix to initialize the weight variable. Must have a suitable dimension

            b_init : np.array, None
                Matrix of biase to initialize biase. Must have a suitable dimension
        """

        input_dim = self.x.shape[1].value
        w_shape = (input_dim, self.size)
        b_shape = (self.size,)

        self.w = variable(w_shape, w_init, self.law_name, self.law_param, "w", tf.float32)
        self.b = variable(b_shape, b_init, self.law_name, self.law_param, "b", tf.float32)

    def _operator(self) -> None:
        """compute the linear operator b + X * W"""

        self.x_out = tf.add(self.b, tf.matmul(self.x, self.w))

    def build(self,
              x: tf.Tensor,
              w_init: np.ndarray = None,
              b_init: np.ndarray = None) -> tf.Tensor:
        """
        Call the build parents method and return the layer output.
        """

        return super().build(x, w_init, b_init)

    def restore(self) -> None:
        """Restore the input, output tensor and all variables."""

        super().restore()
        self.w = get_tf_tensor(name="w")
        self.b = get_tf_tensor(name="b")


class Conv1dLayer(AbstractLayer):

    def __init__(self,
                 filter_width: int,
                 n_filters: int,
                 stride: int = 1,
                 padding: str = "VALID",
                 add_bias: bool = True,
                 act_funct: Union[str, None] = "relu",
                 keep_proba: Union[tf.Tensor, float] = 1.,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 is_training: Union[tf.Tensor, None] = None,
                 name: str = "conv",
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.001,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):
        """
        Build a 1d convolution layer. The filter take as input an array with shape (batch_size, Width, Channel),
        compute a convolution using one or many filter and return a tensor with size (batch_size, new_Width, n_filters).
        The Width can change regarding the filter_width, the stride and the padding selected.

        Args
        ----

            filter_width : int
                Width of the filter apply on the matrix.

            n_filter : int
                Number of filter of the convolution

            stride : int
                Stride for the filter moving

            padding : int
                padding can be SAME or VALID. If PADDING is SAME the convolution output a matrix having the same size
                as the input tensor. VALID keep the dimension reduction.

            add_bias: bool
                Whether to add the biase or not.

            act_funct : str, None
                name of the activation function to use. If None no activation function are used.

            keep_proba : float
                probability to keep a neuron activate during trainning if we want to apply the dropout method.

            batch_norm: bool
                If True apply the batch normalization after the _operator methods.

            batch_renorm: bool
                Whether to used batch renormalization or not.

            is_training : Tensor
                Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

            name : str
                name of the layer

            law_name : str
                name of the lax to use for weights and biases initialization

            law_param : float
                parameter of the initialization law

            decay: float
                Decay used to update the moving average of the batch norm. The moving average is used to learn the
                empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.)

            epsilon: float
                 Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

            decay_renorm: float
                Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

            rmin: Tensor or float
                Minimum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            rmax: Tensor or float
                Maximum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            dmax: Tensor or float
                When batch renormalization is used the scaled mu differences is clipped between (-dmax, dmax)

        Attributes
        ----------

            w : Variable with size (width, n_channel, n_filter)
                weight of the filter

            b : Variable with size (n_filter)
                biase of the convolution

        """

        super().__init__(act_funct, keep_proba, batch_norm, batch_renorm, is_training, law_name, law_param, decay,
                         epsilon, decay_renorm, rmin, rmax, dmax, name)

        assert padding in ["SAME", "VALID"]

        self.filter_width = filter_width
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding
        self.add_bias = add_bias

        self.w: tf.Variable = None
        self.b: tf.Variable = None

    def _check_input(self) -> None:
        """Assert the input tensor have size 3."""

        check_tensor(self.x, shape_dim=3)

    def _init_variable(self, w_init: np.ndarray = None, b_init: np.ndarray = None) -> None:
        """Set all filter variable. Filter can be initialize using outside array.

        Attributes:

            w_init : array with size (width, n_channel, n_filter)
                Array which can be used to initialized weight filter variable

            b_init : array with size (n_filter)
                Array which can be sue dto initialized biase filter variable
        """

        width, n_channel = self.x.shape[1].value, self.x.shape[2].value

        w_shape = (self.filter_width, n_channel, self.n_filters)
        self.w = variable(w_shape, w_init, self.law_name, self.law_param, "w", tf.float32)

        if self.add_bias:
            b_shape = (self.n_filters,)
            self.b = variable(b_shape, b_init, self.law_name, self.law_param, "b", tf.float32)

    def _operator(self) -> None:
        """For simplicity we use the tf.nn.conv1d. According to the tensorflow documentation, this method is just a
          wrapper to tf.nn.conv2d and use a reshape step after and before the conv2d call. Can be improve
          in future version """

        self.x_out = tf.nn.conv1d(self.x, self.w, self.stride, self.padding, data_format="NWC")
        if self.add_bias:
            self.x_out = tf.add(self.b, self.x_out)

    def build(self,
              x: tf.Tensor,
              w_init: np.ndarray = None,
              b_init: np.ndarray = None) -> tf.Tensor:
        """
        Allow to build the convolution taking in entry the x_input tensor.
        """

        return super().build(x, w_init, b_init)

    def restore(self) -> None:
        """Restore all filter variabe an input/output tensor"""

        self.w = get_tf_tensor(name="w")
        if self.add_bias:
            self.b = get_tf_tensor(name="b")
        super().restore()


class MinMaxLayer(AbstractLayer):

    def __init__(self, n_entries: int, name: str = "minmax"):

        """
        This class allow to deployed a MinMax layer. This layer output the top n_entries and the worse n_entries
        of a 2d input tensor.

        Attributes:

            n_entries : int
                number of top and worse neurons to keep

            name: str
                layer name.
        """

        super().__init__(name=name, keep_proba=None, act_funct=None)

        self.n_entries = n_entries

    def _check_input(self) -> None:

        """Assert the input tnput tensor have 2 dimensions."""

        check_tensor(self.x, shape_dim=2)

    def _operator(self) -> None:

        """ The operator use tf.nn.top_k to sort all raw independently from each other then split the tensor into
            2 tensor x_min and x_max and concatenate them to build the final output."""

        input_dim = self.x.shape[1].value

        if input_dim is None:
            x_max = tf.nn.top_k(self.x, k=self.n_entries)[0]
            x_min = -tf.nn.top_k(-self.x, k=self.n_entries)[0]
        else:
            self.x_out = tf.nn.top_k(self.x, k=input_dim)[0]
            x_max, _, x_min = tf.split(
                self.x_out, [self.n_entries, input_dim - 2 * self.n_entries, self.n_entries], axis=1)

        self.x_out = tf.concat([x_min, x_max], axis=1)

    def build(self, x: tf.Tensor) -> tf.Tensor:

        """
        Build the layer
        """

        return super().build(x)

    def restore(self) -> None:

        """Restore only the input nd output tensor using the parents restore method."""

        super().restore()
