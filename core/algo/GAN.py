from typing import Tuple, List, Dict, Any, Sequence, Optional

import numpy as np
import tensorflow as tf

from core.deep_learning.base_architecture import BaseArchitecture
from core.deep_learning.layer import FullyConnected
from core.deep_learning.loss import GanLoss
from core.utils.validation import check_array


class Gan(BaseArchitecture):
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


        x: tf.Tensor, None
            Input tensor of the network.

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

    def __init__(self, name: str = 'BaseGan', use_gpu: bool = False):

        super().__init__(name, use_gpu)

        self.G_layer_size: Tuple = ()
        self.D_layer_size: Tuple = ()
        self.D_act_funct: str = 'relu'
        self.G_act_funct: str = 'relu'
        self.G_final_funct: str = 'tanh'
        self.D_final_funct: str = 'tanh'

        self.input_dim: Optional[int] = None
        self.noise_dim: Optional[int] = None

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
        self.D_optimizer: Optional[tf.Tensor] = None
        self.G_optimizer: Optional[tf.Tensor] = None

        self.D_l_fc: Optional[List[FullyConnected]] = None
        self.D_l_output: Optional[FullyConnected] = None
        self.G_l_fc: Optional[List[FullyConnected]] = None
        self.G_l_output: Optional[FullyConnected] = None
        self.DG_l_fc: Optional[List[FullyConnected]] = None
        self.DG_l_output: Optional[FullyConnected] = None

        self.l_loss: Optional[GanLoss] = None
        self.D_loss: Optional[tf.Tensor] = None
        self.G_loss: Optional[tf.Tensor] = None

    def build(self, D_layer_size: Sequence[int], G_layer_size: Sequence[int], input_dim: int, noise_dim: int,
              D_act_funct: Optional[str] = "relu", G_act_funct: Optional[str] = "relu",
              G_final_funct: Optional[str] = "relu", D_final_funct: str = 'sigmoid', law_name: str = "uniform",
              law_param: float = 0.1, dropout: bool = True, batch_norm: bool = False, batch_renorm: bool = False,
              decay: float = 0.999, decay_renorm: float = 0.99, epsilon: float = 0.001, penalization_rate: float = 0.,
              penalization_type: Optional[str] = None, optimizer_name: str = "Adam") -> None:

        """
        Build the network architecture.

        Args
        ----

            layer_size: Sequence[int]
                Number of neurons for each fully connected step.

            input_dim: int
                Number of input data.

            act_funct: str, None
                Name of the activation function. If None, no activation function is used.

            batch_norm: bool
                If True apply the batch normalization method.

            batch_renorm: bool
                If True apply the batch renormalization method.

            dropout: bool
                Whether to use dropout or not.

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

        super().build(D_layer_size=D_layer_size,
                      G_layer_size=G_layer_size,
                      noise_dim=noise_dim,
                      input_dim=input_dim,
                      D_act_funct=D_act_funct,
                      G_act_funct=G_act_funct,
                      D_final_funct=D_final_funct,
                      G_final_funct=G_final_funct,
                      law_name=law_name,
                      law_param=law_param,
                      dropout=dropout,
                      batch_norm=batch_norm,
                      batch_renorm=batch_renorm,
                      decay=decay,
                      decay_renorm=decay_renorm,
                      epsilon=epsilon,
                      penalization_rate=penalization_rate,
                      penalization_type=penalization_type,
                      optimizer_name=optimizer_name)

    def _build_sub_network(self, x: tf.placeholder, layer_size: Sequence[int], output_dim: int, name: str,
                           act_funct: Optional[str] = 'relu', final_funct: Optional[str] = None,
                           weights: Optional[Sequence[tf.Variable]] = None,
                           bias: Optional[Sequence[tf.Variable]] = None) -> Tuple[List[FullyConnected], FullyConnected]:

        # Define all fully connected layer
        l_fc = []
        i = 0
        x_out = x
        for s in layer_size:
            l_fc.append(
                FullyConnected(size=s,
                               act_funct=act_funct,
                               keep_proba=self.keep_proba,
                               dropout=self.dropout,
                               batch_norm=self.batch_norm,
                               batch_renorm=self.batch_renorm,
                               is_training=self.is_training,
                               name=f"FcLayer_{name}_{i}",
                               law_name=self.law_name,
                               law_param=self.law_param,
                               decay=self.decay,
                               decay_renorm=self.decay_renorm,
                               epsilon=self.epsilon,
                               rmin=self.rmin,
                               rmax=self.rmax,
                               dmax=self.dmax))

            w = None if weights is None else weights[i]
            b = None if bias is None else bias[i]
            x_out = l_fc[-1].build(x_out, w, b)
            i += 1

        # Define the final output layer
        l_output = FullyConnected(size=output_dim,
                                  act_funct=final_funct,
                                  name=f"OutputLayer_{name}",
                                  law_name=self.law_name,
                                  law_param=self.law_param)

        w = None if weights is None else weights[-1]
        b = None if bias is None else bias[-1]
        l_output.build(x_out, w, b)

        return l_fc, l_output

    def _build(self) -> None:

        """Build the MLP Network architecture."""
        super()._build()

        # Define input and noise tensor
        self.x = self._placeholder(tf.float32, (None, self.input_dim), name="x")
        self.z = self._placeholder(tf.float32, (None, self.noise_dim), name="z")

        # Build all Discriminator and Generator layers
        self.D_l_fc, self.D_l_output = self._build_sub_network(self.x, self.D_layer_size, 1, "D",
                                                               act_funct=self.D_act_funct,
                                                               final_funct=self.D_final_funct)

        list_D_var = tf.global_variables()

        self.G_l_fc, self.G_l_output = self._build_sub_network(self.z, self.G_layer_size, self.input_dim, "G",
                                                               act_funct=self.G_act_funct,
                                                               final_funct=self.G_final_funct)

        list_G_var = list(filter(lambda v: v not in list_D_var, tf.global_variables()))

        # Build all D(G) layers
        weights = [l.w for l in self.D_l_fc] + [self.D_l_output.w]
        bias = [l.b for l in self.D_l_fc] + [self.D_l_output.b]
        self.DG_l_fc, self.DG_l_output = self._build_sub_network(self.G_l_output.x_out, self.D_layer_size,
                                                                 1, "DG", act_funct=self.D_act_funct,
                                                                 final_funct=self.D_final_funct,
                                                                 weights=weights, bias=bias)

        # Set the GAN loss function
        self.l_loss = GanLoss(name="GanLoss")
        (self.D_loss,
         self.G_loss) = self.l_loss.build(self.D_l_output.x_out, self.DG_l_output.x_out)

        # Set the optimizer
        self.D_optimizer = self._minimize(self.D_loss, name="D_optimizer", var_list=list_D_var)
        self.G_optimizer = self._minimize(self.G_loss, name="G_optimizer", var_list=list_G_var)

    def fit(self, x: np.ndarray, n_epoch: int = 1, batch_size: int = 10, learning_rate: float = 0.001,
            keep_proba: float = 1., rmax: float = 3., rmin: float = 0.33, dmax: float = 5,
            verbose: bool = True) -> None:

        """ Fit the MLP ``n_epoch`` using the ``x`` and ``y`` array of observations.

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
                Probability to keep a neurone activate during training.

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
        N = len(x)
        n_split = N // batch_size
        sample_index = np.arange(len(x))

        with self.graph.as_default():
            for epoch in range(n_epoch):
                np.random.shuffle(sample_index)
                for batch_index in np.array_split(sample_index, n_split):
                    z = np.random.uniform(-1, 1, (len(batch_index), self.noise_dim))
                    feed_dict = self._get_feed_dict(True, learning_rate, keep_proba, rmin, rmax, dmax)
                    feed_dict.update({self.x: x[batch_index, :],
                                      self.z: z})
                    _, D_loss = self.sess.run([self.D_optimizer, self.D_loss], feed_dict=feed_dict)
                    feed_dict = self._get_feed_dict(True, learning_rate, keep_proba, rmin, rmax, dmax)
                    z = np.random.uniform(-1, 1, (len(batch_index), self.noise_dim))
                    feed_dict.update({self.z: z})
                    _, G_loss = self.sess.run([self.G_optimizer, self.G_loss], feed_dict=feed_dict)
                    self.learning_curve.append((D_loss, G_loss))

                if verbose:
                    print(f'Epoch {epoch}: {self.learning_curve[-1]}')

    def predict(self, N: int) -> np.ndarray:

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

        with self.graph.as_default():
            feed_dict = self._get_feed_dict(is_training=False, keep_proba=1.)
            feed_dict.update({self.z: np.random.uniform(-1, 1, (N, self.noise_dim))})
            return self.sess.run(self.G_l_output.x_out, feed_dict=feed_dict)

    def get_params(self) -> Dict[str, Any]:

        """Get a dictionary containing all network parameters.


        Returns
        -------
            Dict[str, Any]
                Dictionary having all network parameters.

        """

        params = {
            'G_layer_size': self.G_layer_size,
            'D_layer_size': self.D_layer_size,
            'input_dim': self.input_dim,
            'noise_dim': self.noise_dim,
            'D_act_funct': self.D_act_funct,
            'G_act_funct': self.G_act_funct,
            'G_final_funct': self.G_final_funct,
            'D_final_funct': self.D_final_funct,
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
