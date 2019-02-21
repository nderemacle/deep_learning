from abc import ABC, abstractmethod
from typing import Tuple
import tensorflow as tf

from core.deep_learning.tf_utils import get_tf_tensor, get_act_funct
import core.deep_learning.env as env


class AbstractOperator(ABC):

    """Abstract class implementing the build or restore operator.

    The main objective of the framework is to build state of the art deep learning operator where the code
    used to deploy a tensorflow operators is exactly the same than the code used to restore them. Today the framework
    allows to restore a complete graph using a code different to the code used to initialized it. To tackle this problem
    the AbstractOperator define an abstract level which used the build methods like a way to build a first time
    a part of the graph and also a way to restore it. If the build methods is call when the RESTORE environment variable
    is True, all operators of a given graph used to build an algorithm activate their restore methods allowing
    to correctly restore the tensorflow operator. However, if the RESTORE variable is False then the build methods
    call a _build methods which should deploy the part of the graph.

    The main goal is to be more focus on the computation optimization allowing to develop faster more efficient and
    robust algorithms.

    In addition the methods store the name of the algorithm.

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
            self.x_input = tf.identity(x_input, name=f"{self.name}/x_input")
            self.x_output = tf.identity(self.x_input ** (0.5), name=f"{self.name}/x_output")

        def restore(self):
            with tf.get_default_graph() as graph:
                self.x_input = graph.get_tensor_by_name(f"{self.name}/x_input:0")
                self.x_output = graph.get_tensor_by_name(f"{self.name}/x_output:0")

    """

    def __init__(self, name : str):

        self.name = name

    def build(self, *args):

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

        law_name : str (see tf_utils)
            Name of the law to use to initialized Variable.

        law_param : float (see tf_utils)
            Parameter of the law used to initialized weight. This paramter is law_name dependent.

        name : str
            Name of the operator to flag it in the tensorflow graph.

        x_input : Tensor or None
            Input tensor of the operator.

        x_output : Tensor or None
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
                 act_funct : str = None,
                 keep_proba : (tf.Tensor, float, None) = None,
                 law_name : str = "uniform",
                 law_param : float = 0.1,
                 name : str = None):

        super().__init__(name)

        self.x_input : tf.Tensor  = None
        self.x_output : tf.Tensor = None

        self.act_funct = act_funct
        self.keep_proba = keep_proba
        self.law_param = law_param
        self.law_name = law_name

    def _build(self, x_input : tf.Tensor, *init_args):

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

            x_input : Tensor
                Input tensor for the layer.

            init_args: args
                Argument for the weight initialization. Could be an array to initialize Variable values.

        """

        self.x_input = tf.identity(x_input, name=f"{self.name}/input")

        self._check_input()

        self._init_variable(*init_args)

        self._operator()

        if self.act_funct is not None:
            self._apply_act_funct()

        if (self.keep_proba != 1.) & (self.keep_proba is not None):
            self._apply_dropout()

        self.x_output = tf.identity(self.x_output, name=f"{self.name}/output")

    @abstractmethod
    def build(self, *args):

        """ Call the build methods of the parent class. The output of the operator is return allowing to chain
         operators."""

        super().build(*args)

        return self.x_output

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

        self.x_input = get_tf_tensor(name=f"{self.name}/input:0")
        self.x_output = get_tf_tensor(name=f"{self.name}/output:0")

    def _apply_dropout(self):

        """Apply the dropout operator on the output attribute."""

        self.x_output = tf.nn.dropout(self.x_output, keep_prob=self.keep_proba)

    def _apply_act_funct(self):

        """Use an activation function on the output class attribute."""

        act_funct = get_act_funct(self.act_funct)
        self.x_output = act_funct(self.x_output)


class AbstractLoss(AbstractOperator, ABC):

    """Class defining often used attribute and loss when dealing with loss function.

    The class takes advantage of the AbstractOperator class to use the restore or build loss function. The loss can be
    view as a graph operator taking as input a target tensor y and and a network prediction tensor output_network.
    It can use a last transformation or return directly the algorithm prediction y_predict. The output is a loss tensor
    representing the final function to minimize to train the algorithm. In addition regularization can be applied on a
    list of weight transforming the final function to an optimize.

    Attribute:

        penalization_rate : Tensor, float
            Penalization rate for the weight regularization.

        penalization_type : str
            Specify the type of regularization to applied on weight.

        name : str
            Name of the loss operator

        y : Tensor
            Placeholder containing all target variable to learn.

        output_network :  Tensor
            Output of the network.

        y_predict :  Tensor
            Final prediction return by the network.

        loss : Tensor
            loss function of the network

        loss_opt: Tensor
            Final loss function to optimize representing the sum of the loss with all regularization parts.

    Usage:

        class MAE(AbstractLoss):
            def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None):
                super().__init__(penalization_rate, penalization_type, "mae)

            def build(self, output_network : tf.Tensor, y : tf.Tensor, list_weight : Tuple[tf.Variable]=())->tf.Tensor:

                return super().build(y, output_network, list_weight)

            def _set_predict(self):
                self.y_predict = self.output_network

            def _set_loss(self):
                self.loss= tf.reduce_mean(tf.abs(tf.sub(self.y, self.y_predict)))

            def restore(self):
                super().restore()

    """

    def __init__(self, penalization_rate: (tf.Tensor, float) = 0.5, penalization_type: str = None, name : str = "loss"):


        super().__init__(name)

        self.y : tf.Tensor = None
        self.output_network : tf.Tensor = None
        self.y_predict : tf.Tensor = None
        self.loss : tf.Tensor = None
        self.loss_opt : tf.Tensor = None
        self.penality : (float, tf.Tensor) = 0.
        self.penalization_rate = penalization_rate
        self.penalization_type = penalization_type

        self.weights = []

    @abstractmethod
    def build(self, *args) -> (tf.Tensor, tf.Tensor):

        """Use the parents build methods to use the restore or _build process. The build output the loss
           to optimize and the loss function independent of any regularization."""

        super().build(*args)
        return self.loss_opt, self.y_predict

    def check_input(self):

        """Check all input tensor types"""

        assert isinstance(self.y, tf.Tensor)
        assert isinstance(self.output_network, tf.Tensor)
        assert all([isinstance(w, tf.Variable) for w in self.weights])

    @abstractmethod
    def _set_loss(self):

        """Abstract methods which must set the loss tensor using the y_predict and y tensor."""

        raise NotImplementedError()

    @abstractmethod
    def _set_predict(self):


        """Abstract methods which must set the prediction tensor"""
        raise NotImplementedError

    def _build(self, y : tf.Tensor, output_network : tf.Tensor, weights : Tuple[tf.Variable] = (), *args):

        """ The _build method executes the following steps:

        * set all attribute
        * check the format of all tensor input
        * set the loss function
        * set the predict tensor
        * if a list of weight was put in entry add a regularization part to the loss function
        * identify all output tensor

        Attribute:

            y : tf.Tensor
                tensor which contains all objective variable the algorithm learn

            output_network : tf.Tensor
                output of the network which must be transform to obtain the final prediction

            weights : Tuple[Tensor]
                a series of weighs tensor which must be subject to a regularization function.
        """

        self.y = tf.identity(y, name=f"{self.name}/y")
        self.output_network = tf.identity(output_network, name=f"{self.name}/output_network")

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

        self.loss = tf.identity(self.loss, name=f"{self.name}/loss")
        self.loss_opt = tf.identity(self.loss_opt, name=f"{self.name}/loss_opt")
        self.y_predict = tf.identity(self.y_predict, name=f"{self.name}/y_predict")

    @abstractmethod
    def restore(self):

        """Restore all loss tensor attribute"""

        self.loss = get_tf_tensor(name=f"{self.name}/loss:0")
        self.loss_opt = get_tf_tensor(name=f"{self.name}/loss_opt:0")
        self.y_predict = get_tf_tensor(name=f"{self.name}/y_predict:0")
        self.y = get_tf_tensor(name=f"{self.name}/y:0")
        self.output_network = get_tf_tensor(name=f"{self.name}/output_network:0")


    def _compute_penalization(self):

        """ Compute the penalty to apply to a list of weight tensor."""

        if self.penalization_type == 'L2':
            self.penality = tf.add_n([tf.nn.l2_loss(v) for v in self.weights])
        else:
            list_penalization_type = ['L2']
            raise TypeError(
                f"{self.penalization_type} is not a valid method. Method must be in {list_penalization_type}")









