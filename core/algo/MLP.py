from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict, Any, Sequence

import numpy as np
import tensorflow as tf

from core.deep_learning.abstract_architecture import AbstractArchitecture
from core.deep_learning.abstract_operator import AbstractLoss
from core.deep_learning.layer import FcLayer
from core.deep_learning.loss import CrossEntropy, MeanSquareError
from core.utils.validation import check_array


class AbstractMlp(AbstractArchitecture, ABC):
    """
    This class set the major core of a Multi Layer Perceptron neural network. The neural network architecture
    takes as input a linear vector of input data put in a succession of fully connected layers. In the end a
    last layer reduce the dimensionality of the network to match with the number of target variables to predict.

    The abstract schema assume the child class must define the methods to set the loss function. In ths way it is simple
    to enlarge this architecture to any type of problems.

    Args
    ----

        name : str
            Name of the network.

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
            future version.

    Attributes
    ----------

        layer_size: Tuple
            Number of neurons for each fully connected step.

        input_dim: int, None
            Number of input data.

        output_dim: int, None
            Number of target variable to predict.

        act_funct: str, None
            Name of the activation function. If None, no activation function is used.

        keep_proba: float
            Probability to keep a neuron activated during training.

        batch_norm: bool
            If True apply the batch normalization method.

        batch_renorm: bool
            If True apply the batch renormalization method.

        penalization_rate : float
            Penalization rate if regularization is used.

        penalization_type: str, None
            Indicates the type of penalization to use if not None.

        law_name: str
            Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

        law_param: float
            Law parameters dependent to the initialised law choose. If uniform, all tensor
            elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        decay: float
            Decay used to update the moving average of the batch norm. The moving average is used to learn the
            empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

        epsilon: float
            Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

        decay_renorm: float
            Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

        x: tf.Tensor, None
            Input tensor of the network.

        y: tf.Tensor, None
            Tensor containing all True target variable to predict.

        x_out: tf.Tensor, None
            Output of the network.

        loss: tf.Tensor, None
            Loss function optimized to train the MLP.

        y_pred: tf.Tensor, None
            Prediction tensor.

        l_fc: List[FcLayer], None
            List containing all fully connected layer objects.

        l_output: FcLayer, None
            Final layer for network output reduction.

        l_loss: AbstractLoss, None
            Loss layer object.
    """

    def __init__(self, name: str = 'AbstractMlp', use_gpu: bool = False):

        super().__init__(name, use_gpu)

        self.layer_size: Tuple = ()
        self.input_dim: Union[int, None] = None
        self.output_dim: Union[int, None] = None
        self.act_funct: str = 'relu'
        self.keep_proba: float = 1.
        self.batch_norm: bool = False
        self.batch_renorm: bool = False
        self.penalization_rate: float = 0.
        self.penalization_type: Union[str, None] = None
        self.law_name: str = "uniform"
        self.law_param: float = 0.1
        self.decay: float = 0.99
        self.epsilon: float = 0.001
        self.decay_renorm: float = 0.99

        self.x: Union[tf.placeholder, None] = None
        self.y: Union[tf.placeholder, None] = None
        self.x_out: Union[tf.Tensor, None] = None
        self.y_pred: Union[tf.Tensor, None] = None
        self.loss: Union[tf.Tensor, None] = None
        self.optimizer: Union[tf.Tensor, None] = None

        self.l_fc: Union[List[FcLayer], None] = None
        self.l_output: Union[FcLayer, None] = None
        self.l_loss: Union[AbstractLoss, None] = None

    def build(self, layer_size: Sequence[int], input_dim: int, output_dim: int, act_funct: Union[str, None] = "relu",
              keep_proba: float = 1., law_name: str = "uniform", law_param: float = 0.1, batch_norm: bool = False,
              batch_renorm: bool = False, decay: float = 0.999, decay_renorm: float = 0.99, epsilon: float = 0.001,
              penalization_rate: float = 0., penalization_type: Union[str, None] = None,
              optimizer_name: str = "Adam") -> None:

        """
        Build the network architecture.

        Args
        ----

            layer_size: Sequence[int]
                Number of neurons for each fully connected step.

            input_dim: int
                Number of input data.

            output_dim: int
                Number of target variable to predict.

            act_funct: str, None
                Name of the activation function. If None, no activation function is used.

            keep_proba: float
                Probability to keep a neuron activated during training.

            batch_norm: bool
                If True apply the batch normalization method.

            batch_renorm: bool
                If True apply the batch renormalization method.

            penalization_rate : float
                Penalization rate if regularization is used.

            penalization_type: None, str
                Indicates the type of penalization to use if not None.

            law_name: str
                Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

            law_param: float
                Law parameters dependent to the initialised law choose. If uniform, all tensor
                elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
                using a N(0, law_parameters).

            decay: float
                Decay used to update the moving average of the batch norm. The moving average is used to learn the
                empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

            epsilon: float
                Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

            decay_renorm: float
                Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

            optimizer_name: str
                Name of the optimization method use to train the network.

        """

        super().build(layer_size=layer_size,
                      input_dim=input_dim,
                      output_dim=output_dim,
                      act_funct=act_funct,
                      keep_proba=keep_proba,
                      law_name=law_name,
                      law_param=law_param,
                      batch_norm=batch_norm,
                      batch_renorm=batch_renorm,
                      decay=decay,
                      decay_renorm=decay_renorm,
                      epsilon=epsilon,
                      penalization_rate=penalization_rate,
                      penalization_type=penalization_type,
                      optimizer_name=optimizer_name)

    def _build(self) -> None:
        """Build the MLP Network architecture."""

        # Define learning rate and drop_out tensor
        super()._build()

        # Define input and target tensor
        self.x = self._placeholder(tf.float32, (None, self.input_dim), name="x")
        self.y = self._placeholder(tf.float32, (None, self.output_dim), name="y")

        # Define all fully connected layer
        self.l_fc = []
        weights = []
        i = 0
        self.x_out = self.x
        for s in self.layer_size:
            self.l_fc.append(
                FcLayer(size=s,
                        act_funct=self.act_funct,
                        keep_proba=self.keep_proba_tensor,
                        batch_norm=self.batch_norm,
                        batch_renorm=self.batch_renorm,
                        is_training=self.is_training,
                        name=f"FcLayer{i}",
                        law_name=self.law_name,
                        law_param=self.law_param,
                        decay=self.decay,
                        decay_renorm=self.decay_renorm,
                        epsilon=self.epsilon))

            self.x_out = self.l_fc[-1].build(self.x_out)
            weights.append(self.l_fc[-1].w)
            i += 1

        # Define the final output layer
        self.l_output = FcLayer(size=self.output_dim,
                                act_funct=None,
                                keep_proba=1.,
                                name=f"OutputLayer",
                                law_name=self.law_name,
                                law_param=self.law_param)

        self.x_out = self.l_output.build(self.x_out)

        # Set the loss function and the optimizer
        self._set_loss(weights)

    @abstractmethod
    def _set_loss(self, weights: Sequence[tf.Variable]) -> None:

        """This abstract method must set the loss function, the prediction tensor and the optimizer tensor.

        Args

            weights : Sequence[tf.Variable]
                List of weight to apply regularization.

         """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray, n_epoch: int = 1, batch_size: int = 10,
            learning_rate: float = 0.001, rmax: float = 3., rmin: float = 0.33, dmax: float = 5,
            verbose: bool = True) -> None:

        """ Fit the MLP `n_epoch` using the `x` and `y` array of observations.

        Args
        ----

            x: array with shape (n_observation, input_dim)
                Array of input which must have a dimension equal to input_dim.

            y: array with shape (n_observation, output_dim)
                Array of target which must have a dimension equal to output_dim.

            n_epoch: int
                Number of epochs to train the neural network.

            batch_size: int
                Number of observations to used for each backpropagation step.

            learning_rate: float
                Learning rate use for gradient descent methodologies.

            rmin: float
                Minimum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            rmax: float
                Maximum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            dmax: float
                When batch renormalization is used the scaled mu differences is clipped between (-dmax, dmax).

            verbose: bool
                If True print the value of the loss function after each epoch.

        """

        check_array(x, shape=(-1, self.input_dim))
        check_array(y, shape=(-1, self.output_dim))

        sample_index = np.arange(len(x))
        n_split = len(x) // batch_size

        m_loss = 0
        n = 0

        with self.graph.as_default():
            for epoch in range(n_epoch):
                np.random.shuffle(sample_index)
                for batch_index in np.array_split(sample_index, n_split):
                    _, loss = self.sess.run([self.optimizer, self.loss],
                                            feed_dict={self.x: x[batch_index, :],
                                                       self.y: y[batch_index, :],
                                                       self.learning_rate: learning_rate,
                                                       self.keep_proba_tensor: self.keep_proba,
                                                       self.rmin: rmin,
                                                       self.rmax: rmax,
                                                       self.dmax: dmax,
                                                       self.is_training: True})
                    m_loss *= n
                    m_loss += loss
                    n += 1
                    m_loss /= n

                    self.learning_curve.append(loss)

                if verbose:
                    print(f'Epoch {epoch}: {m_loss}')

    def predict(self, x: np.ndarray, batch_size: Union[int, None] = None) -> np.ndarray:

        """
        Make predictions using the `x` array. If `batch_size` is not None predictions are proceed by mini-batch.

        Args
        ----

            x : array with shape (n_observation, input_dim)
                Array of input which must have a dimension equal to input_dim.

            batch_size : int, None
                Number of observations to used for each prediction step. If None predict all label using a single step.

        Returns
        -------
            array with shape (n_observation,)
                Array of predictions
         """

        check_array(x, shape=(-1, self.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            y_predict = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                y_predict.append(self.sess.run(self.y_pred,
                                               feed_dict={self.x: x_batch,
                                                          self.keep_proba_tensor: 1.,
                                                          self.is_training: False}))

            return np.concatenate(y_predict, 0)

    def get_params(self) -> Dict[str, Any]:

        """Get a dictionary containing all network parameters.


        Returns
        -------
            Dict[str, Any]
                Dictionary having all network parameters.

        """

        params = {
            'layer_size': self.layer_size,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'act_funct': self.act_funct,
            'keep_proba': self.keep_proba,
            'batch_norm': self.batch_norm,
            'batch_renorm': self.batch_renorm,
            'law_name': self.law_name,
            'law_param': self.law_param,
            'penalization_rate': self.penalization_rate,
            'decay': self.decay,
            'decay_renorm': self.decay_renorm,
            'epsilon': self.epsilon}

        params.update(super().get_params())

        return params


class MlpClassifier(AbstractMlp):
    """
    This class allows to train a MLP for classification task. The target array must be a One Hot Vector Encoding
    with dimension equal to the number of label to predict. In addition the class provide an additional methods to
    predict directly the probability for each label.

    Args
    ----

        name : str
            Name of the network.

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
            future version.

    Attributes
    ----------

        layer_size: Sequence[int]
            Number of neurons for each fully connected step.

        input_dim: int, None
            Number of input data.

        output_dim: int, None
            Number of target variable to predict.

        act_funct: str, None
            Name of the activation function. If None, no activation function are used.

        keep_proba: float
            Probability to keep a neuron activate during training.

        batch_norm: bool
            If True apply the batch normalization method.

        batch_renorm: bool
            If True apply the batch renormalization method.

        penalization_rate : float
            Penalization rate if regularization is used.

        penalization_type: str, None
            Indicates the type of penalization to use if not None.

        law_name: str
            Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

        law_params: float
            Law parameters dependent to the initialised law choose. If uniform, all tensor
            elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        decay: float
            Decay used to update the moving average of the batch norm. The moving average is used to learn the
            empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

        epsilon: float
            Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

        decay_renorm: float
            Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

        x: tf.Tensor, None
            Input tensor of the network.

        y: tf.Tensor, None
            Tensor containing all True target variable to predict.

        x_out: tf.Tensor, None
            Output of the network.

        loss: tf.Tensor, None
            Loss function optimized to train the MLP.

        y_pred: tf.Tensor, None
            Prediction tensor.

        l_fc: List[FcLayer], None
            List containing all fully connected layer objects.

        l_output: FcLayer, None
            Final layer for network output reduction.

        l_loss: AbstractLoss, None
            Loss layer object.
    """

    def __init__(self, name: str = 'MlpClassifier', use_gpu: bool = False):
        super().__init__(name, use_gpu)

    def _set_loss(self, weights: Sequence[tf.Variable]) -> None:
        """
        Use the cross entropy class to define the network loss function.

        Args
        ----

            weights : Sequence[tf.Variable]
                List of weight to apply regularization.
        """

        self.l_loss = CrossEntropy(penalization_rate=self.penalization_rate,
                                   penalization_type=self.penalization_type,
                                   name=f"cross_entropy")

        self.loss_opt, self.y_pred = self.l_loss.build(y=self.y,
                                                       x_out=self.x_out,
                                                       weights=weights)

        self.loss = self.l_loss.loss

        self.optimizer = self._minimize(self.loss_opt, name="optimizer")

    def predict_proba(self, x: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Predict a vector of probability for each label.

        Args
        ----

            x: array with shape (n_observations, n_inputs)
                Array of input which must have a dimension equal to input_dim.

            batch_size: int
                Number of observation to used for each prediction step. If None predict all label using a single step.

        Returns
        -------

            array with shape (n_observation, n_labels)
                Array of predicted probabilities.
        """

        check_array(x, shape=(-1, self.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            y_pred = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                y_pred.append(self.sess.run(self.x_out,
                                            feed_dict={self.x: x_batch,
                                                       self.keep_proba_tensor: 1.,
                                                       self.is_training: False}))

            y_pred = np.exp(np.concatenate(y_pred, 0))

            return y_pred / y_pred.sum(1).reshape(-1, 1)


class MlpRegressor(AbstractMlp):
    """
    This class allows to train a MLP for regression task. The target array must be a square matrix having one or more
    objective variable to learn.

    Args
    ----

        name : str
            Name of the network.

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
            future version.

    Attributes
    ----------

        layer_size: Sequence[int]
            Number of neurons for each fully connected step.

        input_dim: int, None
            Number of input data.

        output_dim: int, None
            Number of target variable to predict.

        act_funct: str, None
            Name of the activation function. If None, no activation function is used.

        keep_proba: float
            Probability to keep a neuron activated during training.

        batch_norm: bool
            If True apply the batch normalization method.

        batch_renorm: bool
            If True apply the batch renormalization method.

        penalization_rate : float
            Penalization rate if regularization is used.

        penalization_type: str, None
            Indicates the type of penalization to use if not None.

        law_name: str
            Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

        law_param: float
            Law parameters dependent to the initialised law choose. If uniform, all tensor
            elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        decay: float
            Decay used to update the moving average of the batch norm. The moving average is used to learn the
            empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

        epsilon: float
            Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

        decay_renorm: float
            Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

        x: tf.Tensor, None
            Input tensor of the network.

        y: tf.Tensor, None
            Tensor containing all True target variable to predict.

        x_out: tf.Tensor, None
            Output of the network.

        loss: tf.Tensor, None
            Loss function optimized to train the MLP.

        y_pred: tf.Tensor, None
            Prediction tensor.

        l_fc: List[FcLayer], None
            List containing all fully connected layer objects.

        l_output: FcLayer, None
            Final layer for network output reduction.

        l_loss: AbstractLoss, None
            Loss layer object.

    """

    def __init__(self, name: str = 'MlpRegressor', use_gpu: bool = False):
        super().__init__(name, use_gpu)

    def _set_loss(self, weights: Sequence[tf.Variable]) -> None:
        """
        Use the MeanSquareError class to define the network loss function.

        Args
        ----

            weights : Sequence[tf.Variable]
                List of weight to apply regularization.
        """

        self.l_loss = MeanSquareError(penalization_rate=self.penalization_rate,
                                      penalization_type=self.penalization_type,
                                      name=f"mean_square_error")

        self.loss_opt, self.y_pred = self.l_loss.build(y=self.y,
                                                       x_out=self.x_out,
                                                       weights=weights)

        self.loss = self.l_loss.loss

        self.optimizer = self._minimize(self.loss_opt, name="optimizer")
