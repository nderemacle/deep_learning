from typing import Union

import numpy as np
import tensorflow as tf

from core.deep_learning.abstract_operator import AbstractLayer
from core.deep_learning.tf_utils import variable, get_tf_tensor
from core.utils.validation import check_tensor


class FcLayer(AbstractLayer):
    """
    Use a fully connected layer having a number of neurons equal to the `size` parameters. A neurons is a
    function making a linear transformation on a set of input and output a single output value bounded by an
    activation function: :math:`y = \\sigma(b + W \\times x)`

    To prevent overfitting the class allows the used of state of the art deep learning regularization method: batch
    normalization, batch renormalization and dropout.

    Args
    ----

       size: int
           Number of neurons of the layer.

       act_funct: str, None
           name of the activation function to use. If None, no activation function is used.

       keep_proba: (tf.Tensor, float)
           Probability to keep a neuron activated during training.

       batch_norm: bool
            If True apply the batch normalization method.

       batch_renorm: bool
            If True apply the batch renormalization method.

       is_training: tf.Tensor, None
           Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

       name: str
           Name of the layer.

       law_name : str
            Name of the law to use to initialized weights and biases.

       law_param : float
            Parameter of the law used to initialized weight. This parameter is law_name dependent.

       decay: float
            Decay used to update the moving average of the batch norm. The moving average is used to learn the
            empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

       epsilon: float
            Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

       decay_renorm: float
            Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

       rmin: tf.Tensor or float
           Minimum ratio used to clip the standard deviation ratio when batch renormalization is applied.

       rmax: tf.Tensor or float
           Maximum ratio used to clip the standard deviation ratio when batch renormalization is applied.

       dmax: tf.Tensor or float
           When batch renormalization is used the scaled mu differences is clipped between (-dmax, dmax).

    Attributes
    ----------

       w : tf.Variable with size (input_dim, size)
           Weight of the layer. Must be learnt.

       b : tf.Variable with size (size,)
           Bias of the layer. Must be learnt.
    """

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
                 decay_renorm: float = 0.99,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):

        super().__init__(act_funct, keep_proba, batch_norm, batch_renorm, is_training, law_name, law_param, decay,
                         epsilon, decay_renorm, rmin, rmax, dmax, name)

        self.size = size
        self.w: Union[tf.Variable, None] = None
        self.b: Union[tf.Variable, None] = None

    def _check_input(self) -> None:
        """Assert the input tensor is 2 dimensional."""

        check_tensor(self.x, shape_dim=2)

    def _init_variable(self, w_init: np.ndarray = None, b_init: np.ndarray = None) -> None:
        """Initialize the weight and bias. If init matrix are input variable are initialize using them.

        Args
        ----

            w_init : np.array with shape (input_dim, size), None
                Matrix to initialize the weight variable.

            b_init : np.array with shape (size,), None
                Matrix of bias to initialize bias.
        """

        input_dim = self.x.shape[1].value
        w_shape = (input_dim, self.size)
        b_shape = (self.size,)

        self.w = variable(w_shape, w_init, self.law_name, self.law_param, "w", tf.float32)
        self.b = variable(b_shape, b_init, self.law_name, self.law_param, "b", tf.float32)

    def _operator(self) -> None:
        """compute the linear operator :math:`b + W \\times x`."""

        self.x_out = tf.add(self.b, tf.matmul(self.x, self.w))

    def build(self,
              x: tf.Tensor,
              w_init: np.ndarray = None,
              b_init: np.ndarray = None) -> tf.Tensor:
        """
        Call the build parents method and return the layer output.

        Args
        ----

            x: tf.Tensor
                Input array with shape (n_observation, size)

            w_init : np.array with shape (input_dim, size), None
                Matrix to initialize the weight variable.

            b_init : np.array with shape (size,), None
                Matrix of bias to initialize bias.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        return super().build(x, w_init, b_init)

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """

        super().restore()
        self.w = get_tf_tensor(name="w")
        self.b = get_tf_tensor(name="b")


class Conv1dLayer(AbstractLayer):
    """
    Build a 1d convolution layer. The filter take as input an array with shape (batch_size, Width, Channel),
    compute a convolution using one or many filter and return a tensor with size (batch_size, new_Width, n_filters).
    The Width can change regarding the filter_width, the stride and the padding selected.

    Args
    ----

        filter_width : int
            Width of the filter apply on the matrix.

        n_filter : int
            Number of filter of the convolution.

        stride : int
            Stride for the filter moving.

        padding : str
            Padding can be SAME or VALID. If PADDING is SAME the convolution output a matrix having the same size
            as the input tensor. VALID keep the dimension reduction.

        add_bias: bool
            Whether to add the biase or not.

        act_funct : str, None
            Name of the activation function to use. If None no activation function are used.

        keep_proba : float, tf.Tensor
            Probability to keep a neuron activate during training if we want to apply the dropout method.

        batch_norm: bool
            If True apply the batch normalization after the _operator methods.

        batch_renorm: bool
            Whether to used batch renormalization or not.

        is_training : tf.Tensor, None
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        name : str
            Name of the layer.

        law_name : str
            Name of the lax to use for weights and biases initialization.

        law_param : float
            Parameter of the initialization law.

        decay: tf.Tensor, float
            Decay used to update the moving average of the batch norm. The moving average is used to learn the
            empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

        epsilon: float
             Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

        decay_renorm: tf.Tensor, float
            Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

        rmin: tf.Tensor, float
            Minimum ratio used to clip the standard deviation ratio when batch renormalization is applied.

        rmax: tf.Tensor, float
            Maximum ratio used to clip the standard deviation ratio when batch renormalization is applied.

        dmax: tf.Tensor, float
            When batch renormalization is used the scaled mu differences is clipped between (-dmax, dmax).

    Attributes
    ----------

        w : Variable with size (width, n_channel, n_filter)
            Weight of the filter.

        b : Variable with size (n_filter,)
            Bias of the convolution.

    """

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
                 decay_renorm: float = 0.99,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):

        super().__init__(act_funct, keep_proba, batch_norm, batch_renorm, is_training, law_name, law_param, decay,
                         epsilon, decay_renorm, rmin, rmax, dmax, name)

        assert padding in ["SAME", "VALID"]

        self.filter_width = filter_width
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding
        self.add_bias = add_bias

        self.w: Union[tf.Variable, None] = None
        self.b: Union[tf.Variable, None] = None

    def _check_input(self) -> None:
        """Assert the input tensor is 3 dimensional."""

        check_tensor(self.x, shape_dim=3)

    def _init_variable(self, w_init: np.ndarray = None, b_init: np.ndarray = None) -> None:
        """
        Set all filter variable. Filter can be initialize using outside array.

        Args
        ----

            w_init : array with size (width, n_channel, n_filter)
                Array which can be used to initialized weight filter variable.

            b_init : array with size (n_filter,)
                Array which can be sue dto initialized bias filter variable.
        """

        width, n_channel = self.x.shape[1].value, self.x.shape[2].value

        w_shape = (self.filter_width, n_channel, self.n_filters)
        self.w = variable(w_shape, w_init, self.law_name, self.law_param, "w", tf.float32)

        if self.add_bias:
            b_shape = (self.n_filters,)
            self.b = variable(b_shape, b_init, self.law_name, self.law_param, "b", tf.float32)

    def _operator(self) -> None:
        """
        For simplicity we use the tf.nn.conv1d. According to the tensorflow documentation, this method is just a
        wrapper to tf.nn.conv2d and use a reshape step after and before the conv2d call. Can be improve
        in future version
        """

        self.x_out = tf.nn.conv1d(self.x, self.w, self.stride, self.padding, data_format="NWC")
        if self.add_bias:
            self.x_out = tf.add(self.b, self.x_out)

    def build(self,
              x: tf.Tensor,
              w_init: Union[np.ndarray, None] = None,
              b_init: Union[np.ndarray, None] = None) -> tf.Tensor:
        """
        Allow to build the convolution taking in entry the x_input tensor.

        Args
        ----

            x: tf.Tensor
                Input 3 dimensional tensor having the format 'NWC'.

            w_init: np.array with shape (width, n_channel, n_filter), None
                Array to initialize the weight variable.

            b_init : np.array with shape (n_filter,), None
                Array to initialize bias.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        return super().build(x, w_init, b_init)

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """

        self.w = get_tf_tensor(name="w")
        if self.add_bias:
            self.b = get_tf_tensor(name="b")
        super().restore()


class MinMaxLayer(AbstractLayer):
    """
    Allow to use a MinMax layer. Given a 2 dimensional input array, this layer keep only the n best and the n
    worse entries. The aim is to reduce the problem dimensionality by keeping only extremes values from the
    input data.

    Args
    ----

        n_entries : int
            Number of top and worse neurons to keep.

        name: str
            Layer name.
    """

    def __init__(self, n_entries: int, name: str = "minmax"):

        super().__init__(name=name, keep_proba=None, act_funct=None)

        self.n_entries = n_entries

    def _check_input(self) -> None:

        """Assert the input input tensor is 2 dimensional."""

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

    def build(self,
              x: tf.Tensor) -> tf.Tensor:

        """
        Build the MinMax layer using the 2 dimensional input tensor x.

        Args
        ----

            x: tf.Tensor
                Input tensor filtered by the layer.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        return super().build(x)

    def restore(self) -> None:

        """
        Restore input/output tensor.
        """

        super().restore()
