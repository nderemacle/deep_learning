import tensorflow as tf
import numpy as np

from core.deep_learning.tf_utils import build_variable, get_tf_tensor
from core.deep_learning.abstract_operator import AbstractLayer


class FcLayer(AbstractLayer):

    def __init__(self,
                 size : int,
                 act_funct : (str, None) = "relu",
                 keep_proba : (tf.Tensor, float) = 1.,
                 name : str = "fc",
                 law_name : str = "uniform",
                 law_param : float = 0.1):

        """
        Deployed a fully connected layer having a size equal to it number of neurons. It inherit to all trainning
        methods implement in the AbstractLayer class and can used it.

        Attributes:
            size : int,
                number of neurons to output

            act_funct : str
                name of the activation function to use. If None no activation function are used.

            keep_proba : float
                probability to keep a neuron activate during trainning if we want to apply the dropout method.

            name : str
                name of the layer

            law_name : str
                name of the lax to use for weights and biases initialization

            law_param : float
                parameter of the initialization law

            w : Variable with size (input_dim, size)
                Weights of the layer. Must be learnt.

            b : Variable with size (size)
                Biases of the layer. Must be learnt.
        """

        super().__init__(act_funct, keep_proba, law_name, law_param, name)

        self.size = size
        self.w : tf.Variable = None
        self.b : tf.Variable = None

    def _check_input(self):
        """Assert the input tensor have shape 2"""

        assert isinstance(self.x_input, tf.Tensor)
        assert len(self.x_input.shape) == 2

    def _init_variable(self, w_init : np.ndarray = None, b_init : np.ndarray = None):

        """Initialize the weight and biase. If init matrix are input variable are initialize using them.

        Attributes:

            w_int : np.array, None
                Matrix to initialize the weight variable. Must have a suitable dimension

            b_init : np.array, None
                Matrix of biase to initialize biase. Must have a suitable dimension
        """

        input_dim = self.x_input.shape[1].value
        w_shape = (input_dim, self.size)
        b_shape = (self.size,)

        self.w = build_variable(w_shape, w_init, self.law_name, self.law_param, f"{self.name}/w", tf.float32)
        self.b = build_variable(b_shape, b_init, self.law_name, self.law_param, f"{self.name}/b", tf.float32)

    def _operator(self):

        """compute the linear operator b + X * W"""

        self.x_output = tf.add(self.b, tf.matmul(self.x_input, self.w))

    def build(self,
              x_input : tf.Tensor,
              w_init : np.ndarray = None,
              b_init : np.ndarray = None) -> tf.Tensor:

        """
        Call the build parents method and return the layer output.
        """

        return super().build(x_input, w_init, b_init)

    def restore(self):

        """Restore the input, output tensor and all variables."""

        super().restore()
        self.w = get_tf_tensor(name=f"{self.name}/w:0")
        self.b = get_tf_tensor(name=f"{self.name}/b:0")


class Conv1dLayer(AbstractLayer):

    def __init__(self,
                 filter_width : int,
                 n_filters : int,
                 stride : int = 1,
                 padding : str = "VALID",
                 act_funct : (str, None) = "relu",
                 keep_proba : (tf.Tensor, float) = 1.,
                 name : str = "conv",
                 law_name : str = "uniform",
                 law_param : float = 0.1):
        """
        Build a 1d convolution layer. The filter take as input an array with shape (batch_size, Width, Channel),
        compute a convolution using one or many filter and return a tensor with size (batch_size, new_Width, n_filters).
        The Width can change regarding the filter_width, the stride and the padding selected.

        Attributes:

            filter_width : int
                Width of the filter apply on the matrix.

            n_filter : int
                Number of filter of the convolution

            stride : int
                Stride for the filter moving

            padding : int
                padding can be SAME or VALID. If PADDING is SAME the convolution output a matrix having the same size
                as the input tensor. VALID keep the dimension reduction.

            act_funct : str, None
                name of the activation function to use. If None no activation function are used.

            keep_proba : float
                probability to keep a neuron activate during trainning if we want to apply the dropout method.

            name : str
                name of the layer

            law_name : str
                name of the lax to use for weights and biases initialization

            law_param : float
                parameter of the initialization law

            w : Variable with size (width, n_channel, n_filter)
                weight of the filter

            b : Variable with size (n_filter)
                biase of the convolution

        """

        super().__init__(act_funct, keep_proba, law_name, law_param, name)

        assert padding in ["SAME", "VALID"]

        self.filter_width = filter_width
        self.n_filters = n_filters
        self.stride = stride
        self.padding = padding

        self.w : tf.Variable = None
        self.b : tf.Variable = None

    def _check_input(self):

        """Assert the input tensor have size 3."""

        assert isinstance(self.x_input, tf.Tensor)
        assert len(self.x_input.shape) == 3

    def _init_variable(self, w_init : np.ndarray = None, b_init : np.ndarray = None):

        """Set all filter variable. FIlter can be initialize using outside array.

        Attributes:

            w_init : array with size (width, n_channel, n_filter)
                Array which can be used to initialized weight filter variable

            b_init : array with size (n_filter)
                Array which can be sue dto initialized biase filter variable
        """

        width, n_channel = self.x_input.shape[1].value, self.x_input.shape[2].value

        w_shape = (self.filter_width, n_channel, self.n_filters)
        b_shape = (self.n_filters,)
        self.w = build_variable(w_shape, w_init, self.law_name, self.law_param, f"{self.name}/w", tf.float32)
        self.b = build_variable(b_shape, b_init, self.law_name, self.law_param, f"{self.name}/b", tf.float32)

    def _operator(self):

        """For simplicity we use the tf.nn.conv1d. According to the tensorflow documentation, this method is just a
          wrapper to tf.nn.conv2d and use a reshape step after and before the conv2d call. Can be improve
          in future version """

        self.x_output = tf.nn.conv1d(self.x_input, self.w, self.stride, self.padding, data_format="NWC")
        self.x_output = tf.add(self.b, self.x_output)

    def build(self,
              x_input : tf.Tensor,
              w_init : np.ndarray = None,
              b_init : np.ndarray = None) -> tf.Tensor:

        """
        Allow to build the convolution taking in entry the x_input tensor.
        """

        return super().build(x_input, w_init, b_init)

    def restore(self):

        """Restore all filter variabe an input/output tensor"""

        self.w = get_tf_tensor(name=f"{self.name}/w:0")
        self.b = get_tf_tensor(name=f"{self.name}/b:0")
        super().restore()


class MinMaxLayer(AbstractLayer):

    def __init__(self, n_entries : int, name : str = "minmax"):

        """
        This class allow to deployed a MinMaw layer. This layer output the top n_entries and the worse n_entries
        of a 2d input tensor.

        Attributes:

            n_entries : int
                number of top and worse neurons to keep

            name: str
                layer name.
        """

        super().__init__(name=name, keep_proba=None, act_funct=None)

        self.n_entries = n_entries

    def _check_input(self):

        """Assert the input tnput tensor have 2 dimensions."""

        assert isinstance(self.x_input, tf.Tensor)
        assert len(self.x_input.shape) == 2

    def _operator(self):

        """ The operator use tf.nn.top_k to sort all raw independently from each other then split the tensor into
            2 tensor x_min and x_max and concatenate them to build the final output."""

        input_dim = self.x_input.shape[1].value

        if input_dim is  None:
            x_max = tf.nn.top_k(self.x_input, k=self.n_entries)[0]
            x_min = -tf.nn.top_k(-self.x_input, k=self.n_entries)[0]
        else:
            self.x_output = tf.nn.top_k(self.x_input, k=input_dim)[0]
            x_max, _, x_min = tf.split(
                self.x_output, [self.n_entries, input_dim - 2 * self.n_entries, self.n_entries], axis=1)

        self.x_output = tf.concat([x_min, x_max], axis=1)

    def build(self, x_input : tf.Tensor) -> tf.Tensor:

        """
        Buid the layer
        """

        return super().build(x_input)

    def restore(self):

        """Restore only the input nd output tensor using the aprents restore method."""

        super().restore()























