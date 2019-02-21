from typing import Tuple, List
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

from core.deep_learning.abstract_architecture import AbstractArchitecture
from core.deep_learning.abstract_operator import AbstractLoss
from core.deep_learning.layer import FcLayer
from core.deep_learning.loss import CrossEntropy


class AbstractMlp(AbstractArchitecture, ABC):

    """
    This class set the major cortex to build a Multi Layer Perceptron neural networK. . The neural architecture
    takes as input a linear vector of input data put in a succession of layer of neurons. In the end a
    last layer reduce the dimensionality of the network to match with the number of target variables to predict.

    The abstract schema assume the child class must define the methods to set the loss function. In ths way it is simple
    to enlarge this architecture to any type of problems. In addition this abstract class forced the child class
    to define the get_param methods in order to avoid missing additional parameters which could be problematics when
    the network restoration process is run.

    Attributes:

        layer_size : Tuple
            Number of neurons for each fully connected step.

        input_dim : int
            Number of input data.

        output_dim: int
            Number of target variable to predict

        act_funct : str
            Name of the activation function.

        keep_proba : float:
            Probability to keep a neuron activate during training.

        penalization_rate : float
            Penalization rate if regularization is used.

         law_name : str
            Law of the ransom law to used. Must be "normal" for normal law or "uniform" for uniform law.

        law_params : float
            Law parameters which is dependent to the initialised law choose. If uniform, all tensor
            elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        x_input : Tensor
            Input tensor of the network.

        y_target : Tensor
            Tensor containing all True target variable to predict.

        x_output : Tensor
            Output of the network

        loss : Tensor
            loss function optimized to train the mlp.

        y_predict : Tensor
            Prediction tensor.

        l_fc : List[FcLayer]
            List containing all fully connected layer object.

        l_output : FcLayer
            Final layer for network output reduction.

        l_loss : AbstractLoss
            Loss layer object.
    """

    def __init__(self, name : str = 'AbstractMlp', use_gpu : bool = False):

        super().__init__(name, use_gpu)

        self.layer_size : Tuple = ()
        self.input_dim : int = None
        self.output_dim : int = None
        self.act_funct : str = 'relu'
        self.keep_proba : float = 1.
        self.penalization_rate : float = 0.
        self.law_name : str = "uniform"
        self.law_param : float = 0.1

        self.x_input : tf.placeholder = None
        self.y_target : tf.placeholder = None
        self.x_output : tf.Tensor = None
        self.y_predict : tf.Tensor = None
        self.loss : tf.Tensor = None
        self.optimizer : tf.Tensor = None


        self. l_fc : List[FcLayer] = None
        self.l_output : FcLayer = None
        self.l_loss : AbstractLoss = None

    def build(self, layer_size : Tuple, input_dim : int, output_dim : int, act_funct : str = "relu",
              keep_proba : float = 1.,  law_name  : str = "uniform", law_param : float = 0.1,
              penalization_rate : float = 0., optimizer_name : str = "Adam"):

        super().build(layer_size=layer_size,
                      input_dim=input_dim,
                      output_dim=output_dim,
                      act_funct=act_funct,
                      keep_proba=keep_proba,
                      law_name=law_name,
                      law_param=law_param,
                      penalization_rate=penalization_rate,
                      optimizer_name=optimizer_name)

    def _build(self):

        # Define learning rate and drop_out tensor
        super()._build()

        # Define input and target tensor
        self.x_input = self._placeholder(tf.float32, (None, self.input_dim), name=f"{self.name}/x_input")
        self.y_target = self._placeholder(tf.float32, (None, self.output_dim), name=f"{self.name}/y_target")

        # Define all fully connected layer
        self.l_fc = []
        weights = []
        i = 0
        self.x_output = self.x_input
        for s in self.layer_size:
            self.l_fc.append(
                FcLayer(size=s, act_funct = self.act_funct,
                        keep_proba  = self.keep_proba_tensor,
                        name  = f"{self.name}/FcLayer{i}",
                        law_name  = self.law_name,
                        law_param = self.law_param))

            self.x_output = self.l_fc[-1].build(self.x_output)
            weights.append(self.l_fc[-1].w)
            i += 1

        # Define the final output layer
        self.l_output = FcLayer(size=self.output_dim,
                                act_funct = None,
                                keep_proba  = 1.,
                                name  = f"{self.name}/OutputLayer",
                                law_name  = self.law_name,
                                law_param = self.law_param)

        self.x_output = self.l_output.build(self.x_output)

        # Set the loss function and the optimizer
        self._set_loss(weights)

    @abstractmethod
    def _set_loss(self, weights : List[tf.Variable]):

        """This abstract method must set the loss function, the predition tensor and the optimizer tensor.

         Attributes:

            weights : List[tf.Variable]
                List of weight to apply regularization.

         """
        raise NotImplementedError


    def fit(self, x : np.ndarray, y : np.ndarray, n_epoch : int = 1, batch_size : int = 10,
            learning_rate : float = 0.001, verbose : bool = True):

        """ Given an input and a target array, fit the mlp during n_epoch.

        Attributes:

            x : Array with shpe (n_observation, input_dim)
                Array of input which must have a dimension equal to input_dim.

            y : array with shape (n_observation, output_dim)
                Array of target which must have a dimension equal to output_dim.

            n_epoch : int
                Number of epoch to train the neural network.

            batch_size : int
                Number of observation to used for each backpropagation step.

            learning_rate : float
                Learning_rate to apply for training.

            verbose : bool
                If True print the value of the loss function after each epoch.

        """

        self._check_array(x, shape=(-1, self.input_dim))
        self._check_array(y, shape=(-1, self.output_dim))

        sample_index = np.arange(len(x))
        n_split = len(x) // batch_size

        m_loss = 0
        n = 0

        with self.graph.as_default():
            for epoch in range(n_epoch):
                np.random.shuffle(sample_index)
                for batch_index in np.array_split(sample_index, n_split):

                    _, loss = self.sess.run([self.optimizer, self.loss],
                                            feed_dict={self.x_input: x[batch_index, :],
                                                       self.y_target: y[batch_index, :],
                                                       self.learning_rate: learning_rate,
                                                       self.keep_proba_tensor: self.keep_proba})
                    m_loss *= n
                    m_loss += loss
                    n += 1
                    m_loss /= n

                    self.learning_curve.append(loss)

                if verbose:
                    print(f'Epoch {epoch}: {m_loss}')


    def predict(self, x : np.ndarray, batch_size : int = None) -> np.ndarray:

        """Predict a label given an array of input x

         Attributes:

            x : Array with shape (n_observation, input_dim)
                Array of input which must have a dimension equal to input_dim.

            batch_size : int
                Number of observation to used for each prediction step. If None predict all label using a single step.

         Output:
            Array of predictions
         """

        self._check_array(x, shape=(-1, self.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            y_predict = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                y_predict.append(self.sess.run(self.y_predict,
                                               feed_dict={self.x_input: x_batch, self.keep_proba_tensor: 1.}))

            return np.concatenate(y_predict, 0)


    @abstractmethod
    def get_params(self):

        """Get a dictionary containing all network parameters"""

        params = {
            'layer_size': self.layer_size,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'act_funct': self.act_funct,
            'keep_proba': self.keep_proba,
            'law_name': self.law_name,
            'law_param': self.law_param,
            'penalization_rate': self.penalization_rate}

        params.update(super().get_params())

        return params


class MlpClassifier(AbstractMlp):

    """
    This class allows to train a MLP for classification task. The target array must be a One Hot Vector Encoding
    with dimension equal to the number of label to predict. In addition the class provide an additional methods to
    predict directly the probability for each label.

    """

    def __init__(self, name : str = 'MlpClassifier', use_gpu : bool = False):

        super().__init__(name, use_gpu)

    def _set_loss(self, weights : List[tf.Variable]):

        """Use the cross entropy class to define the network loss function."""

        self.l_loss = CrossEntropy(penalization_rate=self.penalization_rate,
                                   penalization_type="L2",
                                   name=f"{self.name}/cross_entropy")

        self.loss_opt, self.y_predict = self.l_loss.build(y=self.y_target,
                                                          output_network=self.x_output,
                                                          weights=weights)

        self.loss = self.l_loss.loss

        self.optimizer = self._minimize(self.loss_opt, name=f"{self.name}/optimizer")

    def predict_proba(self, x: np.ndarray, batch_size: int = None) -> np.ndarray:

        """
        Predict a vector of probability for each label.

        Attributes:

            x : Array with shape (n_observation, input_dim)
                Array of input which must have a dimension equal to input_dim.

            batch_size : int
                Number of observation to used for each prediction step. If None predict all label using a single step.

         Output:
            Array of estimated probability
        """

        self._check_array(x, shape=(-1, self.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            output_predict = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                output_predict.append(self.sess.run(self.x_output,
                                               feed_dict={self.x_input: x_batch, self.keep_proba_tensor: 1.}))

            output_predict = np.exp(np.concatenate(output_predict, 0))

            return output_predict / output_predict.sum(1).reshape(-1, 1)

    def get_params(self):

        return super().get_params()

