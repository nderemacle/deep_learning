from abc import ABC, abstractmethod
from typing import Union, Any, Sequence, Tuple

import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.tf_utils import get_tf_tensor, get_act_funct, identity
from core.utils.validation import check_tensor, check_variable


class AbstractOperator(ABC):
    """Abstract class implementing the build or restore operator concept.

    The main objective of the framework is to build state of the art deep learning operator where the code
    used to deploy Tensorflow operators is exactly the same than the code used to restore them. Today the framework
    allows to restore a complete graph using a code different to the code used to initialized it. To tackle this problem
    the AbstractOperator define an abstract level which used the build methods like a way to build a first time
    a part of the graph and also a way to restore it. If the build methods is call when the RESTORE environment variable
    is True then all operators of a given graph used to build an algorithm run their restore methods when their build
    methods is called in order to correctly restore the Tensorflow operator. However, if the RESTORE variable
    is False then the build methods called a _build methods which must deploy the part of the graph.

    The main goal is to become more focus about the operator optimization allowing to develop faster more efficient and
    robust algorithms.

    In addition the class used the graph scope level to assign a clean name for all tensor created.

    Args
    ----
        name : str
            Name of the operator. Useful to flag all operator attribute in the Tensorflow graph.

    Examples
    --------
        >>> from typing import Union
        >>>
        >>> import tensorflow as tf
        >>>
        >>> from core.deep_learning.abstract_operator import AbstractOperator
        >>> from core.deep_learning.tf_utils import get_tf_tensor
        >>>
        >>> class Sqrt(AbstractOperator):
        ...
        ...     def __init__(self, name: str):
        ...         super().__init__(name)
        ...         self.x_out : Union[tf.Tensor, None] = None
        ...         self.x : Union[tf.Tensor, None] = None
        ...
        ...     def build(self, x: tf.Tensor) -> None:
        ...         super().build(x)
        ...
        ...     def _build(self, x : tf.Tensor) -> None:
        ...         self.x = tf.identity(x, name="x")
        ...         self.x_out = tf.sqrt(self.x, name="x_out")
        ...
        ...     def restore(self) -> None:
        ...         self.x = get_tf_tensor("x")
        ...         self.x_out = get_tf_tensor("x_out")
        ...
    """

    def __init__(self, name: str):

        self.name = name

    def build(self, *args: Any) -> None:

        """
        Build or restore an existing operator using the valid scope name. If the environment variable is set to True the
        `restore` class method is called whereas if False the `_build private` class method is called.

        Args:

            args: Any
                Build arguments defined by the child class.

        """

        # Add '/' allows to insure tensorflow used a valid network operator name when the build is recalled.
        with tf.name_scope(self.name + "/"):
            if env.RESTORE:
                self.restore()
            else:
                self._build(*args)

    @abstractmethod
    def _build(self, *args: Any) -> None:

        """
        Private method which must contains the main code to build the operator.

        Args
        ----

            args: Any
                Build argument which must be define by the child class.

        """

        raise NotImplementedError

    @abstractmethod
    def restore(self, *args: Any) -> None:

        """
        Restore properly all class attribute. Use `core.deep_learning.tf_utils.get_tf_tensor` to safely restore a
        tensor.

        Args
        ----

            args: Any
                Restore argument which must be define by the child class.
        """

        raise NotImplementedError


class AbstractLayer(AbstractOperator, ABC):
    """This class set a cortex for the implementation of a layer.

    An abstract layer is a cortex for any kind of deep learning layers. It inherit of all AbstractOperator properties
    and allows layers to access to often used deep learning methods such that the activation_function, batch
    normalization or dropout.

    Args
    ----

        act_funct : str, None
            Name for the activation function to use.
            If None, no function is applied.

        keep_proba : tf.Tensor, float, None
            Probability to keep a neuron activate during training.

        batch_norm: bool
            If True apply the batch normalization after the _operator methods.

        batch_renorm: bool
            If True used the batch renormalization after the _operator methods.

        is_training : tf.Tensor, None
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        law_name : str
            Name of the law to use to initialized Variable.

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

        name : str, None
            Name of the operator to flag it in the Tensorflow graph.

    Attributes
    ----------

        x : tf.Tensor, None
            Input tensor of the operator.

        x_out : tf.Tensor, None
            Output of the operator.

    Examples
    --------

        >>> import tensorflow as tf
        >>>
        >>> from core.deep_learning.abstract_operator import AbstractLayer
        >>>
        >>> class Sqrt(AbstractLayer):
        ...
        ...     def __init__(self, name: str):
        ...        super().__init__(name=name)
        ...
        ...     def build(self, x: tf.Tensor) -> tf.Tensor:
        ...        return super().build(x)
        ...
        ...     def _operator(self) -> None:
        ...         self.x_out = tf.sqrt(self.x)
        ...
        ...     def restore(self) -> None:
        ...         super().restore()
        ...
    """

    def __init__(self,
                 act_funct: str = None,
                 keep_proba: Union[tf.Tensor, float, None] = None,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 is_training: Union[tf.Tensor, None] = None,
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.001,
                 rmin: Union[tf.Tensor, float] = 0.33,
                 rmax: Union[tf.Tensor, float] = 3,
                 dmax: Union[tf.Tensor, float] = 5,
                 name: Union[str, None] = None):

        super().__init__(name)

        self.act_funct = act_funct
        self.keep_proba = keep_proba
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.batch_renorm = batch_renorm
        self.law_param = law_param
        self.law_name = law_name
        self.decay = decay
        self.epsilon = epsilon
        self.decay_renorm = decay_renorm
        self.rmin = rmin
        self.rmax = rmax
        self.dmax = dmax

        self.x: Union[tf.Tensor, None] = None
        self.x_out: Union[tf.Tensor, None] = None

    def _build(self, x: tf.Tensor, *init_args: Any):

        """
        Regarding parameters set by the child class, the build methods executes step by step the following methods:

            1. store and identify the operator input
            2. check if the input tensor satisfy all class requirement
            3. initialize variable tensor if needed
            4. execute the _operator methods which must set self.x_out using self.x
            5. apply the batch normalization or the batch renormalization
            6. use an activation function if needed
            7. apply dropout if needed
            8. identify the output

        Args
        ----

            x : tf.Tensor
                Input tensor for the layer.

            init_args: Any
                Argument for the weight initialization. Could be an array to initialize Variable values.

        """

        self.x = identity(x, name="x")
        self._check_input()
        self._init_variable(*init_args)
        self._operator()

        if self.batch_norm | self.batch_renorm:
            self._apply_batch_norm()

        if self.act_funct is not None:
            self._apply_act_funct()

        if (self.keep_proba != 1.) & (self.keep_proba is not None):
            self._apply_dropout()

        self.x_out = identity(self.x_out, name="x_out")

    @abstractmethod
    def build(self, *args: Any) -> tf.Tensor:

        """
        Call the build methods of the parent class which run the build or restore process. The output of the operator
        is return allowing to chain operators. This method is an abstract method to oblige the implementation of all
        arguments specification.

        Args
        ----

            args: Any
                Argument for the _build methods which must contain at least the input operator.

        Returns
        -------

            tf.Tensor
                Output operator Tensor.
        """

        super().build(*args)

        return self.x_out

    def _check_input(self) -> None:

        """Apply test on the layer input."""

        pass

    def _init_variable(self, *init_args) -> None:

        """Allows to initialized all layer Variable tensor."""

        pass

    @abstractmethod
    def _operator(self) -> None:

        """
        The main implementation of the operator must be set here. This methods assume the operator takes as input
        the class attribute self.x and write it output on the attribute self.x_out.
        """

        raise NotImplementedError

    @abstractmethod
    def restore(self) -> None:

        """
        Method which restore all class tensor given the operator name and the current graph. The parent class can be
        call to restore standard input and output tensor avoiding code repetition.
        Use `core.deep_learning.tf_utils.get_tf_tensor` to safely restore a tensor.
        """

        self.x = get_tf_tensor(name="x")
        self.x_out = get_tf_tensor(name="x_out")

    def _apply_dropout(self) -> None:

        """Apply the dropout operator on the output attribute."""
        # TODO: Replace keep_proba by skip_proba or dropout_rate
        self.x_out = tf.nn.dropout(self.x_out, rate=1 - self.keep_proba)

    def _apply_batch_norm(self) -> None:
        """
        Apply batch normalization on the output class attribute before the activation function in order to scale the
        layer avoiding vanishing gradient problems. For a mini-batch with size `B` the normalization is:

            * :math:`\\mu = \\frac{1}{B} \\sum_{i=1}^{B} y_{i}`
            * :math:`\\sigma = \\frac{1}{B} \\sum_{i=1}^{B} (y_{i} - \\mu)^2`
            * :math:`\\hat{y} = \\frac{y - \\mu}{\\sqrt{\\sigma + \\epsilon}} \\times \\gamma + \\beta`

        With :math:`\\gamma` and :math:`\\beta` to parameters learn during training to avoid the network to rebuild the
        initial value. The parameter epsilon is set to avoid infinity problem when dividing by the layer standard
        deviation. For inference, the two momentum are learnt online during training using a moving average depending
        to the decay parameter:

            * :math:`\\mu_{t} = \\mu_{t-1} \\times decay + (1 - decay) \\times \\mu`
            * :math:`\\sigma_{t} = \\sigma_{t-1}  \\times decay +  (1 - decay) \\times \\sigma`

        These last parameters are used to normalize inference data. However, using this process the network
        normalization computation is not the same between train and inference sample. Batch renormalization methods
        allows to tackle this problem by adding an intermediary normalization step. For a mini-batch with size `B`
        the renormalization during training becomes:

            * :math:`r = Clip_{(rmin, rmax)} (\\frac{\\sigma}{\\sigma_{t}})`
            * :math:`d = Clip_{(-dmax, dmax)} (\\frac{\\mu - \\mu_{t}}{\\sigma_{t}})`
            * :math:`\\hat{y} = (\\frac{y - \\mu}{\\sqrt{\\sigma + \\epsilon}} \\times r + d)  \\times \\gamma + \\beta`

        Warnings
        --------

            Data format must be 'NC' or 'NHWC'.

        """

        self.x_out = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=self.decay,
            epsilon=self.epsilon,
            center=True,
            scale=True,
            renorm=self.batch_renorm,
            renorm_clipping={'rmin': self.rmin, 'rmax': self.rmax, 'dmax': self.dmax},
            renorm_momentum=self.decay_renorm,
            trainable=True)(self.x_out, training=self.is_training)

    def _apply_act_funct(self) -> None:

        """Use an activation function on the output class attribute."""

        act_funct = get_act_funct(self.act_funct)
        self.x_out = act_funct(self.x_out)


class AbstractLoss(AbstractOperator, ABC):
    """This class set a cortex for the implementation of a loss.

    The class takes inherit all property of the AbstractOperator class to use the restore or build operator process.
    The loss can be view as a graph operator taking as input a target tensor y and and a network prediction tensor
    x_out. It can use a last transformation or return directly the algorithm prediction y_pred. The output is a loss
    tensor representing the final function to minimize to train the algorithm. In addition regularization function can
    be applied on a list of weight transforming the final function to optimize.

    Args
    ----

        penalization_rate: tf.Tensor, float
            Penalization rate for the weight regularization.

        penalization_type: str
            Specify the type of regularization to applied on weight.

        name: str
            Name of the loss operator

    Attributes
    ----------

        y: tf.Tensor
            Placeholder containing all target variable to learn.

        x_out: tf.Tensor
            Output of the network.

        y_pred: tf.Tensor
            Final prediction return by the network.

        loss: tf.Tensor
            loss function of the network

        loss_opt: tf.Tensor
            Final loss function to optimize representing the sum of the loss with all regularization parts.

    Examples
    --------
        >>> from typing import Union, Sequence
        >>>
        >>> import tensorflow as tf
        >>>
        >>> from core.deep_learning.abstract_operator import AbstractLoss
        >>>
        >>> class MAE(AbstractLoss):
        ...
        ...     def __init__(self, penalization_rate: Union[tf.Tensor, float] = 0.5, penalization_type: str = None):
        ...         super().__init__(penalization_rate, penalization_type, "mae")
        ...
        ...     def build(self, x_out : tf.Tensor, y : tf.Tensor, list_weight : Sequence[tf.Variable]=())-> tf.Tensor:
        ...        return super().build(y, x_out, list_weight)
        ...
        ...     def _set_predict(self) -> None:
        ...         self.y_predict = self.output_network
        ...
        ...     def _set_loss(self) -> None:
        ...       self.loss= tf.reduce_mean(tf.abs(tf.sub(self.y, self.y_pred)))
        ...
        ...    def restore(self) -> None:
        ...         super().restore()
        ...

    """

    def __init__(self, penalization_rate: Union[tf.Tensor, float] = 0.5, penalization_type: str = None,
                 name: str = "loss"):

        super().__init__(name)

        self.y: Union[tf.Tensor, None] = None
        self.x_out: Union[tf.Tensor, None] = None
        self.y_pred: Union[tf.Tensor, None] = None
        self.loss: Union[tf.Tensor, None] = None
        self.loss_opt: Union[tf.Tensor, None] = None
        self.penality: Union[float, tf.Tensor] = 0.
        self.penalization_rate = penalization_rate
        self.penalization_type = penalization_type

        self.weights = []

    @abstractmethod
    def build(self, *args: Any) -> Tuple[tf.Tensor, tf.Tensor]:

        """
        Use the parents build methods to use the restore or _build process. The build output the loss to optimize and
        the loss function independent of any regularization.

        Args
        ----

            args: Any
                Key arguments for the _build methods which must be define by the child class.

        Returns
        -------

            tf.Tensor
                Loss tensor to optimize.

            tf.Tensor
                Loss tensor without any regularization terms.
        """

        super().build(*args)

        return self.loss_opt, self.y_pred

    def check_input(self) -> None:

        """Check all input tensor types"""

        check_tensor(self.y)
        check_tensor(self.x_out)
        [check_variable(w) for w in self.weights]

    @abstractmethod
    def _set_loss(self) -> None:

        """Abstract methods which set the loss tensor using the y_pred and y tensor."""

        raise NotImplementedError()

    @abstractmethod
    def _set_predict(self) -> None:

        """Methods which must set the prediction tensor y_pred use to compute prediction."""

        raise NotImplementedError

    def _build(self, y: tf.Tensor, x_out: tf.Tensor, weights: Sequence[tf.Variable] = (), **kwargs: Any) -> None:

        """
        The _build method executes the following steps:

            1. set all attributes
            2. check the format of all tensor input
            3. set the loss function
            4. set the predict tensor
            5. if the weights sequence is not empty, add a regularization term to the loss function
            6. identify all output tensor

        Args
        ----

            y : tf.Tensor
                Tensor which contains all objective variable the algorithm learn.

            x_out : tf.Tensor
                Output of the network which must be transform to obtain the final prediction.

            weights : Sequence[tf.Variable]
                A series of weighs tensor which must be subject to a regularization function.

            kwargs: Any
                Additional parameters which can be define by the child class.
        """

        self.y = identity(y, name="y")

        self.x_out = identity(x_out, name="x_out")

        self.weights = weights

        self.check_input()

        self._set_loss()

        self._set_predict()

        if (self.penalization_type is not None) & (len(self.weights) != 0):
            assert self.loss is not None
            self._compute_penalization()
            self.loss_opt = tf.add(self.loss, tf.multiply(self.penalization_rate, self.penality))
        else:
            self.loss_opt = self.loss

        self.loss = identity(self.loss, name="loss")
        self.loss_opt = identity(self.loss_opt, name="loss_opt")
        self.y_pred = identity(self.y_pred, name="y_pred")

    @abstractmethod
    def restore(self) -> None:

        """Restore all loss tensor attributes."""

        self.loss = get_tf_tensor(name="loss")
        self.loss_opt = get_tf_tensor(name="loss_opt")
        self.y_pred = get_tf_tensor(name="y_pred")
        self.y = get_tf_tensor(name="y")
        self.x_out = get_tf_tensor(name="x_out")

    def _compute_penalization(self) -> None:

        """ Compute the penalty to apply to a list of weight tensor. TODO: Move this function into tf_utils."""

        if self.penalization_type == 'L2':
            self.penality = tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
        elif self.penalization_type == 'L1':
            self.penality = tf.reduce_sum([tf.reduce_sum(tf.abs(v)) for v in self.weights])
        else:
            list_penalization_type = ['L2', 'L1']
            raise TypeError(
                f"{self.penalization_type} is not a valid method. Method must be in {list_penalization_type}")
