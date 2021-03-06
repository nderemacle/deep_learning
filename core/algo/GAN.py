from typing import List, Dict, Any, Sequence, Optional, Union, Callable

import numpy as np
import tensorflow as tf

from core.deep_learning.base_architecture import BaseArchitecture
from core.deep_learning.layer import FullyConnected
from core.deep_learning.loss import GanLoss
from core.utils.validation import check_array


class NeuralNetParams:

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 layer_size: Sequence[int] = (),
                 act_funct: Optional[Union[str, Callable]] = None,
                 final_funct: Optional[Union[str, Callable]] = None,
                 dropout: bool = False,
                 batch_norm: bool = False,
                 batch_renorm: bool = False,
                 law_name: str = "uniform",
                 law_param: float = 0.01,
                 decay: float = 0.99,
                 epsilon: float = 0.001,
                 decay_renorm: float = 0.99):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_size = layer_size
        self.act_funct = act_funct
        self.final_funct = final_funct
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.batch_renorm = batch_renorm
        self.law_name = law_name
        self.law_param = law_param
        self.decay = decay
        self.epsilon = epsilon
        self.decay_renorm = decay_renorm


class NeuralNetStruct:

    def __init__(self):
        self.x: Optional[tf.placeholder] = None
        self.l_fc: Optional[List[FullyConnected]] = None
        self.l_output: Optional[FullyConnected] = None
        self.loss: Optional[tf.Tensor] = None
        self.optimizer: Optional[tf.Tensor] = None


class Gan(BaseArchitecture):
    """
    This class allows to train a Generative Adversary Neural Network. A GAN is composed by two neural network: the
    discriminator and the generator. The generator aims to generates fake input in order to increase the error of the
    discriminator which aims to predict if an input come from the generator or from real data. Both network play against
    each other making at the optimum the generator generate sample such the discriminator is not able to know which
    sample is real or fake.

    Args
    ----

        name : str
            Name of the network.

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism setting can be improve with
            future version.

    Attributes
    ----------

        gen_params: NeuralNetParams, None
            Parameters of the generator

        dis_params: NeuralNetParams, None
            Parameters of the discriminator

        noise_dim: int, None
            Dimension of the nois array.

        gen_network: NeuralNetStruct, None
            Object owning all layer of the generator.

        dis_network: NeuralNetStruct, None
            Object owning all layer of the discriminator.

        l_loss: GanLoss, None
            Loss layer of the gan.


    """

    def __init__(self, name: str = 'BaseGan', use_gpu: bool = False):

        super().__init__(name, use_gpu)

        self.gen_params: Optional[NeuralNetParams] = None
        self.dis_params: Optional[NeuralNetParams] = None
        self.noise_dim: Optional[int] = None

        self.gen_network: Optional[NeuralNetStruct] = None
        self.dis_network: Optional[NeuralNetStruct] = None
        self.dis_gen_network: Optional[NeuralNetStruct] = None

        self.l_loss: Optional[GanLoss] = None

    def build(self,
              input_dim: int,
              noise_dim: int,
              dis_layer_size: Sequence[int] = (100,),
              gen_layer_size: Sequence[int] = (100,),
              dis_act_funct: Optional[Union[str, Callable]] = "relu",
              gen_act_funct: Optional[Union[str, Callable]] = "relu",
              gen_final_funct: Optional[Union[str, Callable]] = "relu",
              dis_final_funct: Optional[Union[str, Callable]] = 'sigmoid',
              dis_law_name: str = "uniform",
              gen_law_name: str = "uniform",
              dis_law_param: float = 0.1,
              gen_law_param: float = 0.1,
              dis_dropout: bool = False,
              gen_dropout: bool = False,
              dis_batch_norm: bool = False,
              gen_batch_norm: bool = False,
              dis_batch_renorm: bool = False,
              gen_batch_renorm: bool = False,
              dis_decay: float = 0.99,
              gen_decay: float = 0.99,
              dis_decay_renorm: float = 0.99,
              gen_decay_renorm: float = 0.99,
              dis_epsilon: float = 0.001,
              gen_epsilon: float = 0.001,
              optimizer_name: str = "Adam") -> None:

        """
        Build the network architecture.

        Args
        ----
            input_dim: int
                DImension of input to generate.

            noise_dim: int
                Size of the array of random noise.

            dis_layer_size: Sequence[int]
                Number of neurons for each fully connected step.

            gen_layer_size: Sequence[int]
                Number of neurons for each fully connected step.

            dis_act_funct: str, None
                Name of the activation function. If None, no activation function is used.

            gen_act_funct: str, None
                Name of the activation function. If None, no activation function is used.

            dis_final_funct: str, None, Callable
                Name or callable object of the function to use for the final layer. If None, no function is used.

            gen_final_funct: str, None, Callable
                Name or callable object of the function to use for the final layer. If None, no function is used.

            dis_law_name: str
                Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

            gen_law_name: str
                Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

            dis_law_param: float
                Law parameters dependent to the initialised law choose. If uniform, all tensor
                elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
                using a N(0, law_parameters).

            gen_law_param: float
                Law parameters dependent to the initialised law choose. If uniform, all tensor
                elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
                using a N(0, law_parameters).

            dis_dropout: bool
                Whether to use dropout or not.

            gen_dropout: bool
                Whether to use dropout or not.

            dis_batch_norm: bool
                If True apply the batch normalization method.

            gen_batch_norm: bool
                If True apply the batch normalization method.

            dis_batch_renorm: bool
                If True apply the batch renormalization method.

            gen_batch_renorm: bool
                If True apply the batch renormalization method.

            dis_decay: float
                Decay used to update the moving average of the batch norm. The moving average is used to learn the
                empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

            gen_decay: float
                Decay used to update the moving average of the batch norm. The moving average is used to learn the
                empirical mean and variance of the output layer. It is recommended to set this value between (0.9, 1.).

            dis_decay_renorm: float
                Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

            gen_decay_renorm: float
                Decay used to update by moving average the mu and sigma parameters when batch renormalization is used.

            dis_epsilon: float
                Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

            gen_epsilon: float
                Parameters used to avoid infinity problem when scaling the output layer during the batch normalization.

            optimizer_name: sr
                Name of the optimizer to use.

        """

        self.gen_params = NeuralNetParams(input_dim=noise_dim,
                                          output_dim=input_dim,
                                          layer_size=gen_layer_size,
                                          act_funct=gen_act_funct,
                                          final_funct=gen_final_funct,
                                          dropout=gen_dropout,
                                          batch_norm=gen_batch_norm,
                                          batch_renorm=gen_batch_renorm,
                                          law_name=gen_law_name,
                                          law_param=gen_law_param,
                                          decay=gen_decay,
                                          epsilon=gen_epsilon,
                                          decay_renorm=gen_decay_renorm)

        self.dis_params = NeuralNetParams(input_dim=input_dim,
                                          output_dim=1,
                                          layer_size=dis_layer_size,
                                          act_funct=dis_act_funct,
                                          final_funct=dis_final_funct,
                                          dropout=dis_dropout,
                                          batch_norm=dis_batch_norm,
                                          batch_renorm=dis_batch_renorm,
                                          law_name=dis_law_name,
                                          law_param=dis_law_param,
                                          decay=dis_decay,
                                          epsilon=dis_epsilon,
                                          decay_renorm=dis_decay_renorm)

        super().build(optimizer_name=optimizer_name, noise_dim=noise_dim)

    def _build_sub_network(self,
                           net_params: NeuralNetParams,
                           net_struct: NeuralNetStruct,
                           name: str,
                           weights: Optional[Sequence[tf.Variable]] = None,
                           bias: Optional[Sequence[tf.Variable]] = None):

        """Build either the discriminator or the generator. If weights or biase are not None use it to initialise
        all layer.
        """

        # Define all fully connected layer
        net_struct.l_fc = []
        i = 0
        x_out = net_struct.x
        for s in net_params.layer_size:
            net_struct.l_fc.append(
                FullyConnected(size=s,
                               act_funct=net_params.act_funct,
                               keep_proba=self.keep_proba,
                               dropout=net_params.dropout,
                               batch_norm=net_params.batch_norm,
                               batch_renorm=net_params.batch_renorm,
                               is_training=self.is_training,
                               name=f"FcLayer_{name}_{i}",
                               law_name=net_params.law_name,
                               law_param=net_params.law_param,
                               decay=net_params.decay,
                               decay_renorm=net_params.decay_renorm,
                               epsilon=net_params.epsilon,
                               rmin=self.rmin,
                               rmax=self.rmax,
                               dmax=self.dmax))

            w = None if weights is None else weights[i]
            b = None if bias is None else bias[i]
            x_out = net_struct.l_fc[-1].build(x_out, w, b)
            i += 1

        # Define the final output layer
        net_struct.l_output = FullyConnected(size=net_params.output_dim,
                                             act_funct=net_params.final_funct,
                                             batch_norm=False,
                                             batch_renorm=False,
                                             keep_proba=None,
                                             name=f"OutputLayer_{name}",
                                             law_name=net_params.law_name,
                                             law_param=net_params.law_param)

        w = None if weights is None else weights[-1]
        b = None if bias is None else bias[-1]
        net_struct.l_output.build(x_out, w, b)

    def _build(self) -> None:

        """Build the Network architecture."""
        super()._build()

        # Build the discriminator
        self.dis_network = NeuralNetStruct()
        self.dis_network.x = self._placeholder(tf.float32, (None, self.dis_params.input_dim), name="x_dis")
        self._build_sub_network(net_params=self.dis_params, net_struct=self.dis_network, name="dis")
        list_dis_var = tf.global_variables()

        # Build the generator
        self.gen_network = NeuralNetStruct()
        self.gen_network.x = self._placeholder(tf.float32, (None, self.gen_params.input_dim), name="x_gen")
        self._build_sub_network(net_params=self.gen_params, net_struct=self.gen_network, name="gen")
        list_gen_var = list(filter(lambda v: v not in list_dis_var, tf.global_variables()))

        # Build the Dis(Gen) network
        self.dis_gen_network = NeuralNetStruct()
        self.dis_gen_network.x = self.gen_network.l_output.x_out
        self._build_sub_network(net_params=self.dis_params, net_struct=self.dis_gen_network, name="dis_gen",
                                weights=[l.w for l in self.dis_network.l_fc] + [self.dis_network.l_output.w],
                                bias=[l.b for l in self.dis_network.l_fc] + [self.dis_network.l_output.b])

        # Set all loss
        self.l_loss = GanLoss(name="GanLoss")
        self.dis_network.loss, self.gen_network.loss = self.l_loss.build(self.dis_network.l_output.x_out,
                                                                         self.dis_gen_network.l_output.x_out)

        # Set all optimizer
        self.dis_network.optimizer = self._minimize(self.dis_network.loss, name="dis_optimizer", var_list=list_dis_var)
        self.gen_network.optimizer = self._minimize(self.gen_network.loss, name="gen_optimizer", var_list=list_gen_var)

    def fit(self, x: np.ndarray, n_epoch: int = 1, batch_size: int = 10, learning_rate: float = 0.001,
            keep_proba: float = 1., rmax: float = 3., rmin: float = 0.33, dmax: float = 5,
            verbose: bool = True) -> None:

        """ Fit the GAN ``n_epoch`` using the ``x`` array of observations.

        Args
        ----

            x: array with shape (n_observation, input_dim)
                Array of input which must have a dimension equal to input_dim.

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

        check_array(x, shape=(-1, self.dis_params.input_dim))
        N = len(x)
        n_split = N // batch_size
        sample_index = np.arange(len(x))
        dis_feed_dict = self._get_feed_dict(True, learning_rate, keep_proba, rmin, rmax, dmax)
        gen_feed_dict = self._get_feed_dict(True, learning_rate, keep_proba, rmin, rmax, dmax)

        with self.graph.as_default():
            for epoch in range(n_epoch):
                np.random.shuffle(sample_index)
                for batch_index in np.array_split(sample_index, n_split):
                    # Update Discriminator
                    z = np.random.uniform(-1, 1, (len(batch_index), self.noise_dim))
                    dis_feed_dict.update({self.dis_network.x: x[batch_index, :], self.gen_network.x: z})
                    _, dis_loss = self.sess.run([self.dis_network.optimizer, self.dis_network.loss],
                                                feed_dict=dis_feed_dict)
                    # Update Generator
                    z = np.random.uniform(-1, 1, (batch_size, self.noise_dim))
                    gen_feed_dict.update({self.gen_network.x: z})
                    _, gen_loss = self.sess.run([self.gen_network.optimizer, self.gen_network.loss],
                                                feed_dict=gen_feed_dict)
                    # Update Learning Curve
                    self.learning_curve.append((dis_loss, gen_loss))

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

        check_array(x, shape=(-1, self.dis_params.input_dim))

        n_split = 1 if batch_size is None else len(x) // batch_size

        with self.graph.as_default():
            feed_dict = self._get_feed_dict(is_training=False, keep_proba=1.)
            y_predict = []
            for x_batch in [x] if batch_size is None else np.array_split(x, n_split, axis=0):
                feed_dict.update({self.dis_network.x: x_batch})
                y_predict.append(self.sess.run(self.dis_network.l_output.x_out, feed_dict=feed_dict))

            return np.concatenate(y_predict, 0)

    def generate(self, n: int) -> np.ndarray:

        """
        Generate random sample using the generator.

        Args
        ----

            n: int
                SIze of the sample to generate.

        Returns
        -------
            array with shape (n_observation,)
                Array of predictions
         """

        with self.graph.as_default():
            feed_dict = self._get_feed_dict(is_training=False, keep_proba=1.)
            feed_dict.update({self.gen_network.x: np.random.uniform(-1, 1, (n, self.noise_dim))})
            return self.sess.run(self.gen_network.l_output.x_out, feed_dict=feed_dict)

    def get_params(self) -> Dict[str, Any]:

        """Get a dictionary containing all network parameters.


        Returns
        -------
            Dict[str, Any]
                Dictionary having all network parameters.

        """

        params = {
            "gen_params": self.gen_params,
            "dis_params": self.dis_params,
            "noise_dim": self.noise_dim,
            "optimizer_name": self.optimizer_name,
        }

        params.update(super().get_params())

        return params
