from abc import abstractmethod
from typing import List, Union, Dict, Any, Sequence, Optional, Callable

import numpy as np
import tensorflow as tf

from core.deep_learning.base_architecture import BaseArchitecture
from core.deep_learning.base_operator import BaseLoss
from core.deep_learning.layer import FullyConnected, Conv2d, Pool2d, Res2d
from core.deep_learning.loss import CrossEntropy, MeanSquareError
from core.utils.validation import check_array


class BaseConvNet(BaseArchitecture):
    """This class set the major core of a Convolution Neural Network (ConvNet).

    The ConvNet takes as input a 3D tensor of input data which are filtered using a series of convolution, pooling and
    inception steps. In the end data are used as input of a series of fully connected layer in order to solve the
    prediction task.

    The abstract schema assume the child class must define the methods to set the loss function. In ths way it is simple
    to enlarge this architecture to many types of problem.

    Args
    ----

        name: str
            Name of the network.

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
            future version.

    Attributes
    ----------
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

        l_conv: List[Union[Conv2d]], None
            List containing all input filter layer.

        l_fc: List[FullyConnected], None
            List containing all fully connected layer objects.

        l_output: FullyConnected, None
            Final layer for network output reduction.

        l_loss: AbstractLoss, None
            Loss layer object.
    """

    def __init__(self, name: str = 'BaseConvNet', use_gpu: bool = False):

        super().__init__(name, use_gpu)

        self.conv_params: Sequence[Dict[str, Any]] = []
        self.fc_size: Sequence[int] = []
        self.input_dim: Optional[Sequence[int]] = None
        self.output_dim: Optional[int] = None
        self.act_funct: Optional[Union[str, Callable]] = "relu"
        self.final_funct: Optional[Union[str, Callable]] = None

        self.dropout: bool = False
        self.batch_norm: bool = False
        self.batch_renorm: bool = False
        self.penalization_rate: float = 0.
        self.penalization_type: Optional[str] = None
        self.law_name: str = "uniform"
        self.law_param: float = 0.1
        self.decay: float = 0.99
        self.epsilon: float = 0.001
        self.decay_renorm: float = 0.99

        self.x: Optional[tf.placeholder] = None
        self.y: Optional[tf.placeholder] = None
        self.x_out: Optional[tf.Tensor] = None
        self.y_pred: Optional[tf.Tensor] = None
        self.loss: Optional[tf.Tensor] = None
        self.optimizer: Optional[tf.Tensor] = None

        self.l_conv: Optional[List[Union[Conv2d]]] = None
        self.l_fc: Optional[List[FullyConnected]] = None
        self.l_output: Optional[FullyConnected] = None
        self.l_loss: Optional[BaseLoss] = None

    def build(self,
              conv_params: Sequence[Dict[str, Any]],
              fc_size: Sequence[int],
              input_dim: Sequence[int],
              output_dim: int,
              act_funct: Optional[Union[str, Callable]] = "relu",
              final_funct: Optional[Union[str, Callable]] = None,
              law_name: str = "uniform",
              law_param: float = 0.1,
              dropout: bool = False,
              batch_norm: bool = False,
              batch_renorm: bool = False,
              decay: float = 0.99,
              decay_renorm: float = 0.99,
              epsilon: float = 0.001,
              penalization_rate: float = 0.,
              penalization_type: Optional[str] = None,
              optimizer_name: str = "Adam") -> None:

        """Build the network architecture.

        The class allows to chains different type of filtering operators ordered inside the ``conv_params`` class
        attributes. Each step can be set using a dictionary of parameters. All dictionary have at least a key ``type``
        informing about the type of step to use: ``CONV``, ``POOL`` or ``RES``. Their parameters are the following:

            * type == ``CONV``:

                * **shape**: (Tuple[int]) - Tuple of filter shape (height, width, filter).
                * **stride**: (Tuple[int]) - Tuple indicating the strides to use (heights, width).
                * **padding**: (str) -  The padding method to use ``SAME`` or ``VALID``.
                * **add_bias**: (bool) -  Use a bias or not.
                * **act_funct**: (None, str) -  Name of the activation function to use. If None, no activation function
                  is used.
                * **dilation**: (None, Tuple[int]) - Tuple indicating the dilation level to use (heights, width).
                  If None no dilation is used.

            * type == ``POOL``:

                * **shape**: (Tuple[int]) - Tuple of pooling shape (height, width, filter).
                * **stride**: (Tuple[int]) - Tuple indicating the strides to use (heights, width).
                * **padding**: (str) -  The padding method to use "SAME" or "VALID".
                * **dilation**: (None, Tuple[int]) - Tuple indicating the dilation level to use (heights, width).
                  If None no dilation is used.
                * **pooling_type**: (str) - Type of pooling to use. Possible values are: ``MAX``, ``MIN``, ``AVG``

            * type == ``RES``:

                * **lag**: (int) - Output lag to consider for residual comparison.
                * **shape**: (Tuple[int], None) - Tuple of pooling reshaping (height, width, filter).
                * **stride**: (Tuple[int], None) - Tuple indicating the strides to use (heights, width).
                * **padding**: (str, None) -  The padding method to use "SAME" or "VALID".
                * **dilation**: (None, Tuple[int]) - Tuple indicating the dilation level to use (heights, width).
                  If None no dilation is used.
                * **pooling_type**: (str, None) - Type of pooling to use. Possible values are: ``MAX``, ``MIN``, ``AVG``

        Warnings
        --------
            Dilation is not available for pooling_type == ``ÀVG``.

        Args
        ----

            conv_params: Sequence[Dict[str, Any]]
                Sequence containing all parameters of all convolution step. See description above for details.

            fc_size: Sequence[int]
                Number of neurons for each fully connected step.

            input_dim: Sequence[int], None
                3D input shape which must be (heights, width, channel).

            output_dim: int, None
                Number of target variable to predict.

            act_funct: str, None
                Name or callable object of the activation function used for the fully connected layer.
                If None, no activation function is used.

            final_funct: str, None, Callable
                Name or callable object of the function to use for the final layer. If None, no function is used.

            dropout: bool
                Whether to use dropout or not.

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

        super().build(conv_params=conv_params,
                      fc_size=fc_size,
                      input_dim=tuple(input_dim),
                      output_dim=output_dim,
                      act_funct=act_funct,
                      final_funct=final_funct,
                      dropout=dropout,
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
        """Build the Convolution Network architecture."""

        # Define learning rate and drop_out tensor
        super()._build()

        # Define input and target tensor
        self.x = self._placeholder(tf.float32, (None,) + tuple(self.input_dim), name="x")
        self.y = self._placeholder(tf.float32, (None, self.output_dim), name="y")

        # Define all fully connected layer
        self.l_conv = []
        weights = []
        self.x_out = self.x
        for param, i in zip(self.conv_params, range(len(self.conv_params))):
            if param["type"] == "CONV":
                self.l_conv.append(
                    Conv2d(width=param["shape"][1],
                           height=param["shape"][0],
                           filter=param["shape"][2],
                           stride=param["stride"],
                           dilation=param["dilation"],
                           padding=param["padding"],
                           add_bias=param["add_bias"],
                           act_funct=param["act_funct"],
                           dropout=self.dropout,
                           batch_norm=self.batch_norm,
                           batch_renorm=self.batch_renorm,
                           is_training=self.is_training,
                           name=f"ConvStep{i}",
                           law_name=self.law_name,
                           law_param=self.law_param,
                           keep_proba=self.keep_proba,
                           decay=self.decay,
                           epsilon=self.epsilon,
                           decay_renorm=self.decay_renorm,
                           rmin=self.rmin,
                           rmax=self.rmax,
                           dmax=self.dmax))

                self.x_out = self.l_conv[-1].build(self.x_out)
                weights.append(self.l_conv[-1].w)

            elif param["type"] == "POOL":
                self.l_conv.append(
                    Pool2d(width=param["shape"][1],
                           height=param["shape"][0],
                           stride=param["stride"],
                           dilation=param["dilation"],
                           padding=param["padding"],
                           pooling_type=param["pooling_type"],
                           name=f"ConvStep{i}"))

                self.x_out = self.l_conv[-1].build(self.x_out)

            elif param["type"] == "RES":
                if len(self.l_conv) - param["lag"] < 0:
                    raise TypeError(f"Too high lag for layer {i}. "
                                    f"Get {param['lag']} whereas {len(self.l_conv)} layers are already set.")
                else:
                    x_lag = self.l_conv[-param["lag"]].x

                width = None if param["shape"] is None else param["shape"][1]
                heigth = None if param["shape"] is None else param["shape"][0]
                self.l_conv.append(
                    Res2d(width=width,
                          height=heigth,
                          stride=param["stride"],
                          padding=param["padding"],
                          dilation=param["dilation"],
                          pooling_type=param["pooling_type"],
                          name=f"ConvStep{i}"))

                self.x_out = self.l_conv[-1].build(self.x_out, x_lag)

            else:
                list_type = ['CONV', 'POOL', 'RES']
                raise TypeError(f"Later type must be in {list_type}")

        # Flatten the current netork state. TODO: Find a solution for build/restore reshaping
        self.x_out = tf.reshape(self.x_out, [-1, int(np.prod(self.x_out.shape[1:]))])

        # Build all fully connected layers
        self.l_fc = []
        for s, i in zip(self.fc_size, range(len(self.fc_size))):
            self.l_fc.append(
                FullyConnected(size=s,
                               act_funct=self.act_funct,
                               dropout=self.dropout,
                               batch_norm=self.batch_norm,
                               batch_renorm=self.batch_renorm,
                               is_training=self.is_training,
                               name=f"FcLayer{i}",
                               law_name=self.law_name,
                               law_param=self.law_param,
                               keep_proba=self.keep_proba,
                               decay=self.decay,
                               decay_renorm=self.decay_renorm,
                               epsilon=self.epsilon,
                               rmin=self.rmin,
                               rmax=self.rmax,
                               dmax=self.dmax))

            self.x_out = self.l_fc[-1].build(self.x_out)
            weights.append(self.l_fc[-1].w)

        # Define the final output layer
        self.l_output = FullyConnected(size=self.output_dim,
                                       act_funct=self.final_funct,
                                       batch_norm=False,
                                       batch_renorm=False,
                                       keep_proba=None,
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
            learning_rate: float = 0.001, keep_proba: float = 1., rmax: float = 3., rmin: float = 0.33, dmax: float = 5,
            verbose: bool = True) -> None:

        """ Fit the ConvNet along  ``n_epoch`` using the ``x`` and ``y`` array of observations.

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

            keep_proba: float
                Probability to keep a neurons activate.

            rmin: float
                Minimum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            rmax: float
                Maximum ratio used to clip the standard deviation ratio when batch renormalization is applied.

            dmax: float
                When batch renormalization is used the scaled mu differences is clipped between (-dmax, dmax).

            verbose: bool
                If True print the value of the loss function after each epoch.

        """

        check_array(x, shape=(-1,) + tuple(self.input_dim))
        check_array(y, shape=(-1, self.output_dim))

        sample_index = np.arange(len(x))
        n_split = len(x) // batch_size

        with self.graph.as_default():
            for epoch in range(n_epoch):
                np.random.shuffle(sample_index)
                for batch_index in np.array_split(sample_index, n_split):
                    feed_dict = self._get_feed_dict(True, learning_rate, keep_proba, rmin, rmax, dmax)
                    feed_dict.update({self.x: x[batch_index, ...], self.y: y[batch_index, ...]})
                    _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                    self.learning_curve.append(loss)

                if verbose:
                    print(f'Epoch {epoch}: {self.learning_curve[-1]}')

    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:

        """
        Make predictions using the ``x`` array. If ``batch_size`` is not None predictions are predicted by mini-batch.

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

        check_array(x, shape=(-1,) + tuple(self.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            y_predict = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                feed_dict = self._get_feed_dict(is_training=False)
                feed_dict.update({self.x: x_batch})
                y_predict.append(self.sess.run(self.y_pred, feed_dict=feed_dict))

            return np.concatenate(y_predict, 0)

    def get_params(self) -> Dict[str, Any]:

        """Get a dictionary containing all network parameters.


        Returns
        -------
            Dict[str, Any]
                Dictionary having all network parameters.

        """

        params = {
            'conv_params': self.conv_params,
            'fc_size': self.fc_size,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'act_funct': self.act_funct,
            'final_funct': self.final_funct,
            'dropout': self.dropout,
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


class ConvNetClassifier(BaseConvNet):
    """This class allow to train a Convolution Network for classification problems.

       The ConvNet takes as input a 3D tensor of input data which are filtered using a series of convolution, pooling and
       inception steps. In the end data are used as input of a series of fully connected layer in order to solve the
       prediction task.

       Args
       ----

           name: str
               Name of the network.

           use_gpu: bool
               If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
               future version.

       Attributes
       ----------

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

           l_fc: List[FullyConnected], None
               List containing all fully connected layer objects.

           l_output: FullyConnected, None
               Final layer for network output reduction.

           l_loss: AbstractLoss, None
               Loss layer object.
    """

    def __init__(self, name: str = 'ConvNetClassifier', use_gpu: bool = False):
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

    def predict_proba(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict a vector of probability for each label. If ``batch_size`` is not None predictions are predicted
        by mini-batch

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

        check_array(x, shape=(-1,) + tuple(self.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            y_pred = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                feed_dict = self._get_feed_dict(is_training=False, keep_proba=1.)
                feed_dict.update({self.x: x_batch})
                y_pred.append(self.sess.run(self.x_out, feed_dict=feed_dict))

            y_pred = np.exp(np.concatenate(y_pred, 0))

            return y_pred / y_pred.sum(1).reshape(-1, 1)


class ConvNetRegressor(BaseConvNet):
    """This class allow to train a Convolution Network for regression problems.

    The ConvNet takes as input a 3D tensor of input data which are filtered using a series of convolution, pooling and
    inception steps. In the end data are used as input of a series of fully connected layer in order to solve the
    prediction task.

    Args
    ----

       name: str
           Name of the network.

       use_gpu: bool
           If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
           future version.

    Attributes
    ----------

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

       l_fc: List[FullyConnected], None
           List containing all fully connected layer objects.

       l_output: FullyConnected, None
           Final layer for network output reduction.

       l_loss: AbstractLoss, None
           Loss layer object.
    """

    def __init__(self, name: str = 'ConvNetRegressor', use_gpu: bool = False):
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
