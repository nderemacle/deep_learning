from typing import Union, Tuple, Optional

import numpy as np
import tensorflow as tf

from core.deep_learning.base_operator import BaseLayer
from core.deep_learning.tf_utils import variable, get_tf_tensor, get_dim_reduction
from core.utils.validation import check_tensor


class FullyConnected(BaseLayer):
    """
    Use a fully connected layer having a number of neurons equal to the `size` parameters. A neurons is a
    function making a linear transformation on a set of input and output a single output value bounded by an
    activation function:

    .. math::

                y = \\sigma(b + W \\times x)

    To prevent overfitting the class allows the used of state of the art deep learning regularization method: batch
    normalization, batch renormalization and dropout.

    Args
    ----

       size: int
           Number of neurons of the layer.

       act_funct: str, None
           name of the activation function to use. If None, no activation function is used.

       dropout: bool
            Whether to use dropout or not.

       batch_norm: bool
            If True apply the batch normalization method.

       batch_renorm: bool
            If True apply the batch renormalization method.

       is_training: tf.Tensor, None
           Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

       name: str
           Name of the layer.

       law_name: str
            Name of the law to use to initialized weights and biases.

       law_param: float
            Parameter of the law used to initialized weight. This parameter is law_name dependent.

       keep_proba: (tf.Tensor, float)
            Probability to keep a neuron activated during training.

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
                 act_funct: Optional[str] = None,
                 dropout: bool = False,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 is_training: Optional[Union[tf.Tensor, bool]] = None,
                 name: str = "fc",
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 keep_proba: Optional[Union[tf.Tensor, float]] = None,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.99,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):

        super().__init__(act_funct, dropout, batch_norm, batch_renorm, is_training, law_name, law_param, keep_proba,
                         decay, epsilon, decay_renorm, rmin, rmax, dmax, name)

        self.size = size
        self.w: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def _check_input(self) -> None:
        """Assert the input tensor is 2 dimensional."""

        check_tensor(self.x, shape_dim=2)

    def _init_variable(self, w: Optional[tf.Variable] = None, b: Optional[tf.Variable] = None) -> None:
        """
        Initialize the weight and bias. If init matrix are input variable are initialize using them.

        Args
        ----

            w : tf.Variable, None
                If not None use this variable as weight.

            b : tf.Variable, None
                If not None use this variable as bias.
        """

        input_dim = self.x.shape[1].value
        w_shape = (input_dim, self.size)
        if w is None:
            self.w = variable(w_shape, None, self.law_name, self.law_param, "w", tf.float32)
        else:
            if w.shape != w_shape:
                raise TypeError("Weight variable must have suitable dimensions regarding input dimension. "
                                f"W shape is {w.shape} whereas x shape is {self.x.shape}")
            self.w = tf.identity(w, name="w")

        if not (self.batch_norm or self.batch_renorm):
            b_shape = (self.size,)
            if b is None:
                    self.b = variable(b_shape, None, self.law_name, self.law_param, "b", tf.float32)
            else:
                if b.shape != b_shape:
                    raise TypeError("Bias variable must have suitable dimensions regarding W shape. "
                                    f"b shape is {b.shape} whereas w shape is {self.w.shape}")
                self.b = tf.identity(b, name="b")

    def _operator(self) -> None:
        """compute the linear operator :math:`b + W \\times x`."""

        self.x_out = tf.matmul(self.x, self.w)
        if not (self.batch_norm | self.batch_renorm):
            self.x_out = tf.add(self.b, self.x_out)

    def build(self,
              x: tf.Tensor,
              w: Optional[tf.Variable] = None,
              b: Optional[tf.Variable] = None) -> tf.Tensor:
        """
        Call the build parents method and return the layer output.

        Args
        ----

            x: tf.Tensor
                Input array with shape (n_observation, size)

            w : tf.Variable, None
                If not None use this variable as weight.

            b : tf.Variable, None
                If not None use this variable as bias.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        return super().build(x, w, b)

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """

        super().restore()
        self.w = get_tf_tensor(name="w")

        if not (self.batch_norm | self.batch_renorm):
            self.b = get_tf_tensor(name="b")


class Conv1d(BaseLayer):
    """
    Build a 1d convolution layer. The filter take as input an array with shape (Width, Channel),
    compute a convolution using one or many filter and return a tensor with size (Width, filters):

    .. math::

                \\sigma(b + W \\otimes x)

    Where :math:`\\otimes` is the convolution operation. The output width size can change regarding the filter_width,
    the stride and the padding selected.

    Args
    ----

        width: int
            Width of the filter apply on the matrix.

        channels: int
            Number of filter of the convolution.

        stride: int
            Stride for the filter moving.

        padding: str
            Padding can be SAME or VALID. If padding is SAME, apply padding to input so that input image gets fully
            covered by filter and stride whereas VALID skip input not covered by the filter.

        add_bias: bool
            Whether to add the bias or not.

        act_funct: str, None
            Name of the activation function to use. If None no activation function are used.

        dropout: bool
            Whether to use dropout or not.

        batch_norm: bool
            If True apply the batch normalization after the _operator methods.

        batch_renorm: bool
            Whether to used batch renormalization or not.

        is_training: tf.Tensor, None
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        name: str
            Name of the layer.

        law_name: str
            Name of the lax to use for weights and biases initialization.

        law_param: float
            Parameter of the initialization law.

        keep_proba: float, tf.Tensor
            Probability to keep a neuron activate during training if we want to apply the dropout method.

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
                 width: int,
                 channels: int,
                 stride: int = 1,
                 padding: str = "VALID",
                 add_bias: bool = True,
                 act_funct: Optional[str] = "relu",
                 dropout: bool = False,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 is_training: Optional[Union[tf.Tensor, bool]] = None,
                 name: str = "conv1D",
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 keep_proba: Optional[Union[tf.Tensor, float]] = None,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.99,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):

        super().__init__(act_funct, dropout, batch_norm, batch_renorm, is_training, law_name, law_param, keep_proba,
                         decay, epsilon, decay_renorm, rmin, rmax, dmax, name)

        assert padding in ["SAME", "VALID"]

        self.width = width
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.add_bias = add_bias

        self.w: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def _check_input(self) -> None:
        """Assert the input tensor is 3 dimensional."""

        check_tensor(self.x, shape_dim=3)

    def _init_variable(self, w: Optional[tf.Variable] = None, b: Optional[tf.Variable] = None) -> None:
        """
        Set all filter variable. Filter can be initialize using outside array.

        Args
        ----

            w : tf.Variable, None
                If not None use this variable as weight.

            b : tf.Variable, None
                If not None use this variable as bias.
        """

        width, n_channel = self.x.shape[1].value, self.x.shape[2].value
        w_shape = (self.width, n_channel, self.channels)

        if w is None:
            self.w = variable(w_shape, None, self.law_name, self.law_param, "w", tf.float32)
        else:
            if w.shape != w_shape:
                raise TypeError("Weight variable must have suitable dimensions regarding input dimension. "
                                f"W shape is {w.shape} whereas x shape is {self.x.shape}")
            self.w = tf.identity(w, name="w")

        if self.add_bias and not (self.batch_norm or self.batch_renorm):
            b_shape = (self.channels,)
            if b is None:
                self.b = variable(b_shape, None, self.law_name, self.law_param, "b", tf.float32)
            else:
                if b.shape != b_shape:
                    raise TypeError("Bias variable must have suitable dimensions regarding W shape. "
                                    f"b shape is {b.shape} whereas w shape is {self.w.shape}")
                self.b = tf.identity(b, name="b")

    def _operator(self) -> None:
        """
        For simplicity we use the tf.nn.conv1d. According to the tensorflow documentation, this method is just a
        wrapper to tf.nn.conv2d and use a reshape step after and before the conv2d call. Can be improve
        in future version
        """

        self.x_out = tf.nn.conv1d(self.x, self.w, self.stride, self.padding, data_format="NWC")
        if (self.add_bias) & (not (self.batch_norm | self.batch_renorm)):
            self.x_out = tf.add(self.b, self.x_out)

    def build(self,
              x: tf.Tensor,
              w: Optional[tf.Variable] = None,
              b: Optional[tf.Variable] = None) -> tf.Tensor:
        """
        Build the convolution taking in entry the x_input tensor.

        Args
        ----

            x: tf.Tensor
                Input 3 dimensional tensor having the format 'NWC'.

            w : tf.Variable, None
                If not None use this variable as weight.

            b : tf.Variable, None
                If not None use this variable as bias.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        return super().build(x, w, b)

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """

        self.w = get_tf_tensor(name="w")
        if self.add_bias and not (self.batch_norm | self.batch_renorm):
            self.b = get_tf_tensor(name="b")
        super().restore()


class MinMax(BaseLayer):
    """
    Allow to use a MinMax layer. Given a 2 dimensional input array, this layer keep only the n best and the n
    worse entries. The aim is to reduce the problem dimensionality by keeping only extremes values from the
    input layer data.

    Args
    ----

        n_entries: int
            Number of top and worse neurons to keep.

        name: str
            Layer name.
    """

    def __init__(self, n_entries: int, name: str = "minmax"):

        super().__init__(name=name)

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


class Conv2d(BaseLayer):
    """
    Build a two dimensional convolution layer. A convolution is an operation which aims to extract information locally
    from an input tensor with two dimension and many channels. The size of the filter can be define by setting its width
    and height whereas the channel define the number of filter which composed the layer. The move of the filter is
    controlled by adjustment of the stride level. The dilation parameter allows to extract information from
    input spaced by a given rate. Given a filter weight :math:`W` with shape (h, w, c, f) and an input subset
    :math:`x` with shape (h, w, c), the convolution operation is:

    .. math::

                o[i, j, f] = \\sum_{h} \\sum_{w}  \\sum_{c} x[i -h, j - w, c] * W[h, w, c, f]

    The output width size can change regarding the filter_width, the stride and the padding selected and the padding
    level necessary:

    .. math::

                \\cfrac{A - F - (F - 1 )D)}{S} + 1

    where :math:`A` represent the size of an input axis, :math:`F` the filter size, :math:`D` the dilation rate and
    :math:`S` the stride use.

    Args
    ----

        width: int
            Width of the filter.

        height: int
            height of the filter.

        channel: int
            Number of filter of the convolution.

        stride: Tuple[int, int]
            Stride for the filter moving. The first element is the height stride whereas the second is the width stride.

        dilation: Tuple[int, int]
            Number of positions skip between two filter entries. The first element is the height dilation whereas the
            second is the width dilation. If None, no dilation is used.

        padding: str
            Padding can be SAME or VALID. If padding is SAME, apply padding to input so that input image gets fully
            covered by filter and stride whereas VALID skip input not covered by the filter.

        add_bias: bool
            Whether to add the bias or not.

        act_funct: str, None
            Name of the activation function to use. If None no activation function are used.

        dropout: bool
            Whether to use dropout or not.

        batch_norm: bool
            If True apply the batch normalization after the _operator methods.

        batch_renorm: bool
            Whether to used batch renormalization or not.

        is_training: tf.Tensor, None
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        name: str
            Name of the layer.

        law_name: str
            Name of the lax to use for weights and biases initialization.

        law_param: float
            Parameter of the initialization law.

         keep_proba: float, tf.Tensor
            Probability to keep a neuron activate during training if we want to apply the dropout method.

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

        w : Variable with size (height, width, input_channel, n_filter)
            Convolution weights variable.

        b : Variable with size (n_filter,)
            Convolution bias variable.


    """

    def __init__(self,
                 width: int,
                 height: int,
                 filter: int,
                 stride: Tuple[int, int] = (1, 1),
                 dilation: Optional[Tuple[int, int]] = None,
                 padding: str = "VALID",
                 add_bias: bool = False,
                 act_funct: Optional[str] = None,
                 dropout: bool = False,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 is_training: Optional[Union[tf.Tensor, bool]] = None,
                 name: str = "conv2D",
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 keep_proba: Optional[Union[tf.Tensor, bool]] = None,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.99,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5):

        super().__init__(act_funct, dropout, batch_norm, batch_renorm, is_training, law_name, law_param, keep_proba,
                         decay, epsilon, decay_renorm, rmin, rmax, dmax, name)

        self.width = width
        self.height = height
        self.filter = filter
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.add_bias = add_bias

        self.w: Optional[tf.Variable] = None
        self.b: Optional[tf.Variable] = None

    def _check_input(self) -> None:

        """Assert the input input tensor is 4 dimensional."""

        check_tensor(self.x, shape_dim=4)

        if self.x.shape[1] < self.height:
            raise (f"{self.name}: Input heights must be higher or equal to filter heights."
                   f"Input heights {self.x.shape[1]}, filter heights {self.height}")

        if self.x.shape[2] < self.width:
            raise (f"{self.name}: Input heights must be higher or equal to filter heights."
                   f"Input heights {self.x.shape[2]}, filter heights {self.width}")

    def _init_variable(self, w: Optional[tf.Variable] = None, b: Optional[tf.Variable] = None) -> None:
        """
        Initialize the weight and bias. If init matrix are input variable are initialize using them.

        Args
        ----

            w : tf.Variable, None
                If not None use this variable as weight.

            b : tf.Variable, None
                If not None use this variable as bias.
        """

        input_channel = self.x.shape[3].value
        w_shape = (self.height, self.width, input_channel, self.filter)

        if w is None:
            self.w = variable(w_shape, None, self.law_name, self.law_param, "w", tf.float32)
        else:
            if w.shape != w_shape:
                raise TypeError("Weight variable must have suitable dimensions regarding input dimension. "
                                f"W shape is {w.shape} whereas x shape is {self.x.shape}")
            self.w = tf.identity(w, name="w")

        if self.add_bias and not (self.batch_norm or self.batch_renorm):
            b_shape = (self.filter,)
            if b is None:
                self.b = variable(b_shape, None, self.law_name, self.law_param, "b", tf.float32)
            else:
                if b.shape != b_shape:
                    raise TypeError("Bias variable must have suitable dimensions regarding W shape. "
                                    f"b shape is {b.shape} whereas w shape is {self.w.shape}")
                self.b = tf.identity(b, name="b")

    def build(self, x: tf.Tensor, w: Optional[tf.Variable] = None, b: Optional[tf.Variable] = None) -> tf.Tensor:

        """
        Build the Conv2d layer using the 4 dimensional input tensor x.

        Args
        ----

            x: tf.Tensor
                Input tensor filtered by the layer.

            w : tf.Variable, None
                If not None use this variable as weight.

            b : tf.Variable, None
                If not None use this variable as bias.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        return super().build(x, w, b)

    def _operator(self) -> None:

        if self.dilation is None:
            dilation = (1, 1, 1, 1)
        else:
            dilation = (1, self.dilation[0] + 1, self.dilation[1] + 1, 1)

        self.x_out = tf.nn.conv2d(
            input=self.x,
            filter=self.w,
            strides=(1, self.stride[0], self.stride[1], 1),
            padding=self.padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilation)

        if (self.add_bias) & (not (self.batch_norm | self.batch_renorm)):
            self.x_out = tf.add(self.b, self.x_out)

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """

        self.w = get_tf_tensor(name="w")
        if self.add_bias and not (self.batch_norm or self.batch_renorm):
            self.b = get_tf_tensor(name="b")
        super().restore()


class Pool2d(BaseLayer):
    """
    Build a two dimensional pooling layer. The pooling is an operator which aims to reduce the dimension of an image
    without loss of relevant information. It can be view as a filter making aggregation of input at a local level.
    In addition it allows to decrease noise and keep only the most relevant signals in the 2D input matrix.
    During a convolution step the pooling operator allows the neural network to be invariant to short image
    translation which increases the final accuracy.

    The class allows to use the operator min, max and average.

    Warnings
    --------
        Dilation is not available for pooling_type == ``ÀVG``.

    Args
    ----

        width: int
            Width of the pooling.

        height: int
            height of the pooling.

        stride: Tuple[int, int]
            Stride for the pooling moving. The first element is the height stride whereas the second is the width stride.

        padding: str
            Padding can be SAME or VALID. If padding is SAME, apply padding to input so that input image gets fully
            covered by filter and stride whereas VALID skip input not covered by the filter.

        dilation: Tuple[int, int]
            Number of positions skip between two filter entries. The first element is the height dilation whereas the
            second is the width dilation. If None, no dilation is used.

        pooling_type: str
            Pooling operator to use. Must be MAX, MIN or AVG.

        name: str
            Name of the layer.


    """

    def __init__(self,
                 width: int,
                 height: int,
                 stride: Tuple[int, int] = (1, 1),
                 padding: str = "VALID",
                 dilation: Optional[Tuple[int, int]] = None,
                 pooling_type: str = "MAX",
                 name: str = "Pool2D"):

        super().__init__(name=name)

        self.width = width
        self.height = height
        self.stride = stride
        self.padding = padding
        self.pooling_type = pooling_type
        self.dilation = dilation

    def _check_input(self) -> None:

        """Assert the input input tensor is 4 dimensional."""

        check_tensor(self.x, shape_dim=4)

        if self.x.shape[1] < self.height:
            raise (f"{self.name}: Input heights must be higher or equal to filter heights."
                   f"Input heights {self.x.shape[1]}, filter heights {self.height}")

        if self.x.shape[2] < self.width:
            raise (f"{self.name}: Input heights must be higher or equal to filter heights."
                   f"Input heights {self.x.shape[2]}, filter heights {self.width}")

    def build(self,
              x: tf.Tensor) -> tf.Tensor:

        """
        Build the MaxPool2d layer using the 4 dimensional input tensor x.

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

    def _operator(self) -> None:

        ksize = (1, self.height, self.width, 1)
        strides = (1, self.stride[0], self.stride[1], 1)
        data_format = "NHWC"

        self.x_out = tf.multiply(-1., self.x) if self.pooling_type == "MIN" else self.x

        if self.pooling_type in ["MAX", "MIN"]:
            if self.dilation is None:
                self.x_out = tf.nn.max_pool(value=self.x_out, ksize=ksize, strides=strides, padding=self.padding,
                                            data_format=data_format)
            else:
                dilation = (1, self.dilation[0] + 1, self.dilation[1] + 1, 1)
                filter = np.zeros((self.height, self.width, self.x.shape[-1]))
                self.x_out = tf.nn.dilation2d(input=self.x_out, filter=filter, strides=strides, rates=dilation,
                                              padding=self.padding)
        elif self.pooling_type == "AVG":
            if self.dilation is None:
                self.x_out = tf.nn.avg_pool(value=self.x_out, ksize=ksize, strides=strides, padding=self.padding,
                                            data_format=data_format)
            else:
                raise TypeError("Dilation not implemented for pooling_type == 'AVG'.")
        else:
            list_type = ["MAX", "AVG", "MIN"]
            raise TypeError(f"{self.pooling_type} isn't a valid pooling type. Pooling_type must be in {list_type}.")

        self.x_out = tf.multiply(-1., self.x_out) if self.pooling_type == "MIN" else self.x_out

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """
        super().restore()


class Res2d(BaseLayer):
    """Build a residual layer.

    This layer takes input the previous output tensor and the input of an older layer. If :math:`f` represent all
    operation done from the older layer input :math:`x`, the operator computed is the following:

        .. math::

                f(x) + x

    This formalization constrain the network to search an image noise correction of the original input `x` using all
    operator define in the function :math:`f`. This residual correction allows next layer to considers less noisy input
    in order to make the inference task.

    The old input can have dimensions not similar to the residual part. To counter this a pooling step is done to
    reduce the 2 first dimensions whereas a zero padding is applied on the channels dimension. These points make the
    assumption the residual part must have a number of channels higher or equal to the number of channel of the input.
    However the height and the width of the residual parts must be lower or equal to thus of the input tensor. Pooling
    and padding are applied only if these conditions are not reach.

    Args
    ----

        width: int, None
            Width of the pooling to apply if dimension reduction is needed.

        height: int, None
            Height of the pooling to apply if dimension reduction is needed.

        stride: Tuple[int, int]
            Stride for the filter moving. The first element is the height stride whereas the second is the width stride.

        padding: str
            Padding can be SAME or VALID. If padding is SAME, apply padding to input so that input image gets fully
            covered by filter and stride whereas VALID skip input not covered by the filter.

        dilation: Tuple[int, int]
            Number of positions skip between two filter entries. The first element is the height dilation whereas the
            second is the width dilation. If None, no dilation is used.

        pooling_type: str
            Pooling operator to use. Must be MAX, MIN or AVG.

        name: str
            Name of the layer.

    Attributes
    ----------

        x_lag: tf.Tensor
            Input tensor of a previous layer.

        pool: Pool2d
            Pooling layer used to reduce the old input tensor.



    """

    def __init__(self,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 stride: Optional[Tuple[int, int]] = None,
                 padding: Optional[str] = None,
                 dilation: Optional[Tuple[int, int]] = None,
                 pooling_type: Optional[str] = None,
                 name: str = "Residual") -> None:

        super().__init__(name=name)

        self.x_lag: Union[tf.Tensor, None] = None
        self.pool = Pool2d(width, height, stride, padding, dilation, pooling_type, self.name + "/pool")

    def build(self, x: tf.Tensor, x_lag: tf.Tensor) -> tf.Tensor:

        """
        Build the Res2d layer using the 4 dimensional input tensor x.

        Args
        ----

            x: tf.Tensor
                Original tensor.

            x_lag: tf.Tensor
                Transformed tensor.

        Returns
        -------

            tf.Tensor
                Layer output Tensor.
        """

        self.x_lag = x_lag
        return super().build(x)

    def _check_input(self) -> None:
        """Check if both input respect all dimension conditions."""

        check_tensor(self.x, shape_dim=4)
        check_tensor(self.x_lag, shape_dim=4)

        if self.x.shape[3] < self.x_lag.shape[3]:
            raise TypeError(f"{self.name}: x must have a number of channel higher or equal than x_lag. "
                            f"Channel for x is {self.x.shape[3].value} whereas is {self.x_lag.shape[3].value} "
                            "for x_lag.")

        if self.x.shape[1] > self.x_lag.shape[1]:
            raise TypeError(f"{self.name}: x must have a height lower or equal than x_lag. "
                            f"Height for x is {self.x.shape[1].value} whereas is {self.x_lag.shape[1].value}"
                            " for x_lag.")

        elif self.x.shape[1] < self.x_lag.shape[1]:
            if any([x is None for x in [self.pool.height, self.pool.stride, self.pool.padding]]):
                raise TypeError("x_lag reduction needed. You must feed value for height, stride and padding. ")
            dilation = 0 if self.pool.dilation is None else self.pool.dilation[0]
            new_dim = get_dim_reduction(self.x_lag.shape[1].value, self.pool.height, dilation, self.pool.stride[0],
                                        self.pool.padding)

            if new_dim != self.x.shape[1]:
                raise TypeError(f"{self.name}: After transformation height of x must be equal than x_lag. "
                                f"Height for x_lag becomes {new_dim} whereas is {self.x.shape[1].value} for x.")

        if self.x.shape[2] > self.x_lag.shape[2]:
            raise TypeError(f"{self.name}: x must have width lower or equal than x_lag. "
                            f"Width for x is {self.x.shape[2].value} whereas is {self.x_lag.shape[2].value} for x_lag.")

        elif self.x.shape[2] < self.x_lag.shape[2]:
            if any([x is None for x in [self.pool.width, self.pool.stride, self.pool.padding]]):
                raise TypeError("x_lag reduction needed. You must feed value for width, stride and padding. ")
            dilation = 0 if self.pool.dilation is None else self.pool.dilation[1]
            new_dim = get_dim_reduction(self.x_lag.shape[2].value, self.pool.width, dilation, self.pool.stride[1],
                                        self.pool.padding)

            if new_dim != self.x.shape[2]:
                raise TypeError(f"{self.name}: After transformation width of x must be equal than x_lag. "
                                f"Width for x_lag becomes {new_dim} whereas is {self.x.shape[2].value} for x.")

    def _operator(self) -> None:
        """
        Apply the pooling layer on the past input layer if needed and apply a zero padding on it channel dimension.
        Finally compute the residual layer by summing the residual parts with the transform old input layer.
        """

        if self.pool.width is not None:
            self.x_out = self.pool.build(self.x_lag)
        else:
            self.x_out = self.x_lag

        if self.x.shape[-1] > self.x_lag.shape[-1]:
            delta = int(self.x.shape[-1]) - int(self.x_lag.shape[-1])
            self.x_out = tf.pad(self.x_out, [[0, 0], [0, 0], [0, 0], [delta, 0]])

        elif self.x.shape[-1] < self.x_lag.shape[-1]:
                raise TypeError("Channel dimension of x must be higher or equal to channel dimension of x_lag. "
                                f"Dimension of x is {self.x.shape[-1]} whereas is {self.x_lag.shape[-1]} for x_lag.")

        self.x_out = tf.add(self.x, self.x_out)

    def restore(self) -> None:
        """
        Restore input/output tensor and all layer variables.
        """
        self.pool.restore()
        super().restore()
