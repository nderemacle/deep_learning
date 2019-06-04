import os
import traceback
from abc import abstractmethod
from typing import Any, Dict, Union, Sequence, Optional

import numpy as np
import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.tf_utils import get_tf_operation, get_tf_tensor, get_optimizer
from core.utils.reader_writer import write_pickle, read_pickle
from core.utils.validation import is_not_in_graph


class BaseArchitecture:
    """
    Base for any deep learning architecture. The class initialization step allows to configure the Tensorflow
    interface to use GPU computation for example. In addition this abstract level implement general usage such the
    saving and the restoration methods. It provide also some always use methods such the minimizer or placeholder
    setting.

    Args
    ----

        name: str
            Name of the network.

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism setting will be improve with
            future version.

    Attributes
    ----------

        graph: tf.Graph
            Graph of the network useful to isolate network environment if many architecture object are used.

        sess: tf.Session
            Tensorflow session mostly used to make Tensor computations.

        optimizer_name: str
            Name of the optimizer to use. Could become child attribute in future versions

        learning_curve: list
            A list containing the value of the loss after each training step.

        learning_rate: tf.Tensor
            Learnig rate tensor for optimization.

        keep_proba: tf.Tensor
            Tensor for dropout methods.

        is_training: tf.Tensor
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        dmax: tf.Tensor
            Tensor used to clip the batch renormalization scale parameter.

        rmin: tf.Tensor
            Lower bound used to clip the batch renormalization shift parameter.

        rmax: tf.Tensor
            Upper bound used to clip the batch renormalization shift parameter.
    """

    def __init__(self, name: str, use_gpu: bool = False):

        self.name = name
        self.use_gpu = use_gpu
        self.graph = tf.Graph()
        self.sess = self._set_session()

        self.optimizer_name = "Adam"
        self.learning_curve = []

        self.learning_rate: Optional[tf.placeholder] = None
        self.keep_proba: Optional[tf.placeholder] = None
        self.is_training: Optional[tf.placeholder] = None
        self.rmax: Optional[tf.placeholder] = None
        self.rmin: Optional[tf.placeholder] = None
        self.dmax: Optional[tf.placeholder] = None

    def _set_session(self) -> tf.Session:
        """
        Set the Tensorflow Session and return the session object.

        Returns
        -------
            tf.Session
                Tensorflow session object.
        """
        if self.use_gpu:
            conf = tf.ConfigProto(log_device_placement=True,
                                  allow_soft_placement=True,
                                  gpu_options=tf.GPUOptions(allow_growth=True))
        else:
            conf = tf.ConfigProto(allow_soft_placement=True)

        return tf.Session(graph=self.graph, config=conf)

    @abstractmethod
    def _build(self) -> None:

        """Can be called to instance often used deep learning tensor."""

        self.learning_rate = self._placeholder(tf.float32, None, name="learning_rate")
        self.keep_proba = self._placeholder(tf.float32, None, name="keep_proba_tensor")
        self.is_training = self._placeholder(tf.bool, None, name="is_training")
        self.rmax = self._placeholder(tf.float32, 1, name="rmax")
        self.rmin = self._placeholder(tf.float32, 1, name="rmin")
        self.dmax = self._placeholder(tf.float32, 1, name="dmax")

    def build(self, **kwargs: Any) -> None:

        """
         Methods to build the Neural Network cortex. In a first time all network arguments are update into the
         dict class then the graph is build and all variable initialized. The name scope is used to ensure a valid
         tensor naming: network_name/operator_name/tensor. The '/' allows to insure tensorflow used a good network
         name when the build is recalled.


         Args
         ----
            kwargs: Any
                Neural network parameters.
         """

        self.__dict__.update(kwargs)
        with self.graph.as_default():
            with tf.name_scope(self.name + "/"):
                self._build()
                self.sess.run(tf.initializers.global_variables())

    def _get_feed_dict(self, is_training: bool, learning_rate: float = 1, keep_proba: float = 1., rmin: float = 1.,
                       rmax: float = 1., dmax: float = 0.) -> Dict[tf.Tensor, Any]:
        """
        Return a initialize feed_dict with all network parameters correctly set.

        Args
        ----

            is_training: bool
                Indicated whether the feedict is in training or in inference.

            learning_rate: bool
                Learning_rate to use duting training.

            keep_proba: bool
                Probability to let a neurons activate.

            dmax: float
                Tensor used to clip the batch renormalization scale parameter.

            rmin: float
                Lower bound used to clip the batch renormalization shift parameter.

            rmax: float
                Upper bound used to clip the batch renormalization shift parameter.

        Returns
        -------

            Dict[tf.Tensor, Any]
                Feed dictionary with all neural network tensor correctly set.

        """

        return {self.keep_proba: keep_proba, self.rmin: (rmin,), self.rmax: (rmax,), self.dmax: (dmax,),
                self.is_training: is_training, self.learning_rate: learning_rate}

    @abstractmethod
    def fit(self, **kwargs: Any) -> None:

        """
        The fit methods to train the neural network.

        Args
        ----
            kwargs: Any
                Input array and fit parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs: Any) -> np.ndarray:

        """
        Make a prediction using input arrays with shape (n_observations, ...).

        Args
        ----
            kwargs: Any
                Input array and predict parameters.

        Returns
        -------

            array with shape (n_observations, ...)
                Array of prediction.

        """
        raise NotImplementedError

    def save(self, path_folder: str) -> None:

        """
        Save all Tensorflow graph and all network parameters in a folder. The methods use the Tensorflow saver method
        and save all network parameters inside a pickle file.

        Args
        ----

            path_folder : str
                Path of the folder where the network is saved. It must ended by '/'.
        """

        assert path_folder.endswith("/")

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path_folder + "archi.ckpt", global_step=0)

        write_pickle(self.get_params(), path_folder + "param.pkl")

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:

        """
        Return all network parameters. The child class can call the parents get_params methods to get all global
        network parameters.

        Returns
        -------

            Dict[str, Any]
                Dictionary having all network parameters.

        """

        return {
            "optimizer_name": self.optimizer_name,
            "learning_curve": self.learning_curve,
            'name': self.name}

    def _check_and_restore(self, path_folder: str) -> None:

        """
        Check if all folder and file exist and restore the graph and all class attributes then.

        Args
        ----

            path_folder: str
                Path of the folder where the network is saved. It must ended by '/'.
        """

        assert path_folder.endswith("/")

        if not os.path.exists(path_folder):
            raise TypeError(f"{path_folder} does not exist.")

        path = path_folder + "param.pkl"
        if not os.path.exists(path):
            raise TypeError(f"{path} does not exist.")

        self.__dict__.update(read_pickle(path))

        path = path_folder + 'archi.ckpt-0.meta'
        if not os.path.exists(path):
            raise TypeError(f"{path} does not exist.")

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(path, clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(path_folder))

    def restore(self, path_folder: str) -> None:

        """
        Restore the graph by activating all restoration process. The RESTORE environment is first set to True
        to activate all class restoration methods when build is call. If a failure append, set the RESTORE
        environment variable to False and raise the error.

        Use the scope name to use the algorithm name for all tensor and operation of the network.

        Args
        ----

            path_folder : str
                Path of the folder where the network is saved. It must ended by '/'.
        """

        tf.reset_default_graph()
        self._check_and_restore(path_folder)

        env.RESTORE = True

        try:
            with self.graph.as_default():
                with tf.name_scope(self.name):
                    self._build()
        except Exception:
            raise TypeError(traceback.format_exc())
        finally:
            env.RESTORE = False

    def _minimize(self, f: tf.Tensor, name: str = "optimizer") -> Union[tf.train.Optimizer, tf.Tensor]:

        """
        Return an optimizer which minimize a tensor f. Add all parameters store in the UPDATE_OPS such that all moving
        mean and variance parameters of a batch normalization.

        Args
        ----

            f : tf.Tensor
                function to minimize.

            name : str
                name of the tensor optimizer.

        Returns
        -------

            tf.train.Optimizer
                Tensorflow optimizer object.
        """

        if env.RESTORE:
            return get_tf_operation(name, self.graph)
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                return get_optimizer(self.optimizer_name, self.learning_rate).minimize(f, name=name)

    def _placeholder(self, dtype: tf.DType, shape: Union[Sequence[Union[int, None]], int, None],
                     name: str) -> tf.placeholder:

        """
        Set or restore a placeholder.

        Args
        ----

            dtype : tf.DType
                Type of the placeholder.

            shape : Sequence[int], int, None
                Size of the placeholder.

            name : str
                name of the placeholder.

        Returns
        -------

            tf.placeholder
                Tensorflow placeholder object.

        """

        if env.RESTORE:
            return get_tf_tensor(name, self.graph)
        else:
            is_not_in_graph(name, self.graph)
            if shape is None:
                return tf.placeholder(dtype, name=name)
            else:
                return tf.placeholder(dtype, shape, name)
