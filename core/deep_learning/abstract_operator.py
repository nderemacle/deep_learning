from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.tf_utils import get_tf_tensor, get_act_funct, identity
from core.utils.validation import check_tensor, check_variable


class AbstractOperator(ABC):
    """Abstract class implementing the build or restore operator.

    The main objective of the framework is to build state of the art deep learning operator where the code
    used to deploy Tensorflow operators is exactly the same than the code used to restore them. Today the framework
    allows to restore a complete graph using a code different to the code used to initialized it. To tackle this problem
    the AbstractOperator define an abstract level which used the build methods like a way to build a first time
    a part of the graph and also a way to restore it. If the build methods is call when the RESTORE environment variable
    is True, all operators of a given graph used to build an algorithm activate their restore methods allowing
    to correctly restore the Tensorflow operator. However, if the RESTORE variable is False then the build methods
    call a _build methods which should deploy the part of the graph.

    The main goal is to be more focus on the computation optimization allowing to develop faster more efficient and
    robust algorithms.

    In addition the methods used the graph scope level to properly assign the name path of all tensor created.

    Attributes:
        name : str
            Name of the operator. Useful to flag all operator attribute on the tensorflow graph.

    Usage:

    class Sqrt(AbstractOperator):
       def __init__(self, name : str):
            super().__init__(name)
            self.x_output : tf.Tensor = None
            self.x_input : tf.Tensor = None

        def build(self, x_input : tf.Tensor):
            super().build(x_input)

        def _build(self, x_input : tf.Tensor):
            self.x_input = tf.identity(x_input, name="x_input")
            self.x_output = tf.identity(self.x_input ** (0.5), name="x_output")

        def restore(self):
            with tf.get_default_graph() as graph:
                self.x_input = graph.get_tensor_by_name("x_input")
                self.x_output = graph.get_tensor_by_name("x_output")

    """

    def __init__(self, name: str):

        self.name = name

    def build(self, *args):

        """
        Build or restore and existing operator using the valid scope name. The '/' allows to insure tensorflow used
        a valid network operator name when the build is recalled."""

        with tf.name_scope(self.name + "/"):
            if env.RESTORE:
                self.restore()
            else:
                self._build(*args)

    @abstractmethod
    def _build(self, *args):

        """This method must contains the main code of the operator."""

        raise NotImplementedError

    @abstractmethod
    def restore(self, *args):

        """The restore method must restore properly all class tensorflow tensor."""

        raise NotImplementedError


class AbstractLayer(AbstractOperator, ABC):
    """This class allow to generalize the development of a layer.

    An abstract layer represent an abstract shema for any kind of deep learning layers. It takes advantage of all
    AbstractOperator implementation and allows layers to have automatically access to often used application such that
    the activation_function, the batch normalization or the dropout methods.

    Attributes:

        act_funct : str or None (see tf_utils)
            Name for the activation function to use.
            If None, no function is applied.

        keep_proba : tf.Tensor, float or None
            Probability between 0. and 1. to keep activate a neurons during trainning when using the dropout method.
            When a prediction is done we must set this probability to 1. to get a valid prediction. Set a tensor
            allow then to switch from training to prediction. If no dropout is espected set the proba to 1. or None.

        batch_norm: bool
            If True apply the batch normalization after the _operator methods.

        is_training : Tensor
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        law_name : str (see tf_utils)
            Name of the law to use to initialized Variable.

        law_param : float (see tf_utils)
            Parameter of the law used to initialized weight. This parameter is law_name dependent.

        decay: float
            Decay used to update the moving average of the batch norm. The moving average is used to learn the
            empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.)

        epsilon: float
            Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

        name : str
            Name of the operator to flag it in the tensorflow graph.

        x_input : Tensor or None
            Input tensor of the operator.

        x_out : Tensor or None
            Output of the operator.

    Usage:

    class Sqrt(AbstractLayer):
       def __init__(self, name : str):
            super().__init__(name=name)

       def build(self, x_input : tf.Tensor) -> tf.Tensor:
            return super().build(x_input)

       def _operator(self):
            self.x_output = tf.sqrt(self.x_input)

       def restore(self):
            super().restore()
    """

    def __init__(self,
                 act_funct: str = None,
                 keep_proba: (tf.Tensor, float, None) = None,
                 batch_norm: bool = False,
                 is_training: (tf.Tensor, None) = None,
                 law_name: str = "uniform",
                 law_param: float = 0.1,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 name: str = None):

        super().__init__(name)

        self.x: tf.Tensor = None
        self.x_out: tf.Tensor = None

        self.act_funct = act_funct
        self.keep_proba = keep_proba
        self.is_training = is_training
        self.batch_norm = batch_norm
        self.law_param = law_param
        self.law_name = law_name
        self.decay = decay
        self.epsilon = epsilon

    def _build(self, x: tf.Tensor, *init_args):

        """
        Regarding parameters set by the child class, the build methods executes step by step the following methods:

            * store and identify the operator input
            * verify if the input tensor satisfy all class requirement
            * initialize variable tensor if needed
            * execute the _operator methods which must take as input self.x_input and output self.x_output
            * use an activation function in needed
            * apply dropout if needed
            * identify the output

        Attributes:

            x : Tensor
                Input tensor for the layer.

            init_args: args
                Argument for the weight initialization. Could be an array to initialize Variable values.

        """

        self.x = identity(x, name="x")

        self._check_input()

        self._init_variable(*init_args)

        self._operator()

        if self.batch_norm:
            self._apply_batch_norm()

        if self.act_funct is not None:
            self._apply_act_funct()

        if (self.keep_proba != 1.) & (self.keep_proba is not None):
            self._apply_dropout()

        self.x_out = identity(self.x_out, name="x_out")

    @abstractmethod
    def build(self, *args):

        """ Call the build methods of the parent class. The output of the operator is return allowing to chain
         operators."""

        super().build(*args)

        return self.x_out

    def _check_input(self):

        """Allows to define a way to assert the input format"""

        pass

    def _init_variable(self, *init_args):

        """Allows to initialized all useful Variable tensor."""

        pass

    @abstractmethod
    def _operator(self):

        """The main implementation of the operator must be set here. This methods assume the operator takes as input
        the class attribute self.x_input and write it output on the attribute sel.x_output."""

        raise NotImplementedError

    @abstractmethod
    def restore(self):

        """Method which restore all class tensor given the operator name and the current graph. The parent class can be
        call to restore standard input and output tensor avoiding code repetition."""

        self.x = get_tf_tensor(name="x")
        self.x_out = get_tf_tensor(name="x_out")

    def _apply_dropout(self):

        """Apply the dropout operator on the output attribute."""

        self.x_out = tf.nn.dropout(self.x_out, keep_prob=self.keep_proba)

    def _apply_batch_norm(self):
        """
        Apply batch normalization before the activation function in order to scale the layer avoiding vanishing
        gradient problems.

        During training the batch normalization center and scale the output layer using the mean and the variance
        of the layer over the population. These momentum are learn online during training using a moving average with
        the decay parameter: mu_t = decay * mu_t + (1 - decay) * mu_t-1
        The parameter epsilon is set to avoid in problems when dividing by the layer standard deviation.

        """

        self.x_out = tf.contrib.layers.batch_norm(
            inputs=self.x_out,
            decay=self.decay,
            epsilon=self.epsilon,
            updates_collections=tf.GraphKeys.UPDATE_OPS,
            is_training=self.is_training,
            data_format='NHWC',
            renorm=False,
            renorm_decay=self.decay)

    def _apply_act_funct(self):

        """Use an activation function on the output class attribute."""

        act_funct = get_act_funct(self.act_funct)
        self.x_out = act_funct(self.x_out)


class AbstractLoss(AbstractOperator, ABC):
    """Class defining often used attribute and loss when dealing with loss function.

    The class takes advantage of the AbstractOperator class to use the restore or build loss function. The loss can be
    view as a graph operator taking as input a target tensor y and and a network prediction tensor output_network.
    It can use a last transformation or return directly the algorithm prediction y_predict. The output is a loss tensor
    representing the final function to minimize to train the algorithm. In addition regularization can be applied on a
    list of weight transforming the final function to an optimize.

    Attributes:

        penalization_rate : Tensor, float
            Penalization rate for the weight regularization.

        penalization_type : str
            Specify the type of regularization to applied on weight.

        name : str
            Name of the loss operator

        y : Tensor
            Placeholder containing all target variable to learn.

        x_out :  Tensor
            Output of the network.

        y_pred :  Tensor
            Final prediction return by the network.

        loss : Tensor
            loss function of the network

        loss_opt: Tensor
            Final loss function to optimize representing the sum of the loss with all regularization parts.

    Usage:

        class MAE(AbstractLoss):
            def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None):
                super().__init__(penalization_rate, penalization_type, "mae)

            def build(self, x_out : tf.Tensor, y : tf.Tensor, list_weight : Tuple[tf.Variable]=())->tf.Tensor:

                return super().build(y, x_out, list_weight)

            def _set_predict(self):
                self.y_predict = self.output_network

            def _set_loss(self):
                self.loss= tf.reduce_mean(tf.abs(tf.sub(self.y, self.y_pred)))

            def restore(self):
                super().restore()

    """

    def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None, name: str = "loss"):

        super().__init__(name)

        self.y: tf.Tensor = None
        self.x_out: tf.Tensor = None
        self.y_pred: tf.Tensor = None
        self.loss: tf.Tensor = None
        self.loss_opt: tf.Tensor = None
        self.penality: (float, tf.Tensor) = 0.
        self.penalization_rate = penalization_rate
        self.penalization_type = penalization_type

        self.weights = []

    @abstractmethod
    def build(self, *args) -> (tf.Tensor, tf.Tensor):

        """Use the parents build methods to use the restore or _build process. The build output the loss
           to optimize and the loss function independent of any regularization."""

        super().build(*args)
        return self.loss_opt, self.y_pred

    def check_input(self):

        """Check all input tensor types"""

        check_tensor(self.y)
        check_tensor(self.x_out)
        [check_variable(w) for w in self.weights]

    @abstractmethod
    def _set_loss(self):

        """Abstract methods which must set the loss tensor using the y_predict and y tensor."""

        raise NotImplementedError()

    @abstractmethod
    def _set_predict(self):

        """Abstract methods which must set the prediction tensor"""
        raise NotImplementedError

    def _build(self, y: tf.Tensor, x_out: tf.Tensor, weights: Tuple[tf.Variable] = (), *args):

        """ The _build method executes the following steps:

        * set all attribute
        * check the format of all tensor input
        * set the loss function
        * set the predict tensor
        * if a list of weight was put in entry add a regularization part to the loss function
        * identify all output tensor

        Attributes:

            y : tf.Tensor
                tensor which contains all objective variable the algorithm learn

            x_out : tf.Tensor
                output of the network which must be transform to obtain the final prediction

            weights : Tuple[Tensor]
                a series of weighs tensor which must be subject to a regularization function.
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
    def restore(self):

        """Restore all loss tensor attribute"""

        self.loss = get_tf_tensor(name="loss")
        self.loss_opt = get_tf_tensor(name="loss_opt")
        self.y_pred = get_tf_tensor(name="y_pred")
        self.y = get_tf_tensor(name="y")
        self.x_out = get_tf_tensor(name="x_out")

    def _compute_penalization(self):

        """ Compute the penalty to apply to a list of weight tensor."""

        if self.penalization_type == 'L2':
            self.penality = tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
        elif self.penalization_type == 'L1':
            self.penality = tf.reduce_sum([tf.reduce_sum(tf.abs(v)) for v in self.weights])
        else:
            list_penalization_type = ['L2', 'L1']
            raise TypeError(
                f"{self.penalization_type} is not a valid method. Method must be in {list_penalization_type}")
