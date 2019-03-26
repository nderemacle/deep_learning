import os
import traceback
from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf

import core.deep_learning.env as env
from core.deep_learning.tf_utils import get_tf_operation, get_tf_tensor
from core.utils.reader_writer import write_pickle, read_pickle
from core.utils.validation import is_not_in_graph


class AbstractArchitecture(ABC):
    """
    This abstract class defines the cortex for a deep learning architecture. It allows to well configure the
    Tensorflow session and to launch the build or restore process. In addition it give access to functional methods
    such that optimizer setting.

    Args
    ----

        name : str
            name of the network

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism settign can be improve with
            future version

    Attributes
    ----------

        graph : tf.Graph
            graph of the network useful to well define new tensor and restore them.

        sess : tf.Session
            tensorflow session useful for all interaction

        optimizer_name: str
            Name of the optimizer to use. Could become child attribute in futur versions

        learning_curve : list
            a list containing the value of the loss after each training step.

        learning_rate : Tensor
            Learnig rate tensor for optimization.

        keep_proba_tensor : Tensor
            Tensor for dropout methods

        is_training : Tensor
            Tensor indicating if data are used for training or to make prediction. Useful for batch normalization.

        dmax: Tensor
            Tensor used to clip the batch renormalization scale parameter.

        rmin: Tensor
            Lower bound used to clip the batch renormalization shift parameter.

        rmax: Tensor
            Upper bound used to clip the batch renormalization shift parameter.
    """

    def __init__(self, name: str, use_gpu: bool = False):

        self.name = name
        self.use_gpu = use_gpu
        self.graph = tf.Graph()
        self.sess = self._set_session()
        self.optimizer_name = "Adam"
        self.learning_curve = []
        self.learning_rate: tf.placeholder = None
        self.keep_proba_tensor: tf.placeholder = None
        self.is_training: tf.placeholder = None
        self.rmax: tf.placeholder = None
        self.rmin: tf.placeholder = None
        self.dmax: tf.placeholder = None

    def _set_session(self) -> tf.Session:
        """ configure tensorflow graph and session """
        if self.use_gpu:
            conf = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True,
                                  gpu_options=tf.GPUOptions(allow_growth=True))
        else:
            conf = tf.ConfigProto(allow_soft_placement=True)

        return tf.Session(graph=self.graph, config=conf)

    @abstractmethod
    def _build(self):

        """Can be called to instance often used deep learning tensor"""

        self.learning_rate = self._placeholder(tf.float32, None, name="learning_rate")
        self.keep_proba_tensor = self._placeholder(tf.float32, None, name="keep_proba_tensor")
        self.is_training = self._placeholder(tf.bool, None, name="is_training")
        self.rmax = self._placeholder(tf.float32, None, name="rmax")
        self.rmin = self._placeholder(tf.float32, None, name="rmin")
        self.dmax = self._placeholder(tf.float32, None, name="dmax")

    def build(self, **args):

        """ Methods to build the Neural Network cortex. In a first time all network arguments are update into the
         dict class then the graph is build and all variable initialized. The name scope is used to ensure a correct
         tensor naming: network_name/operator_name/tensor. The '/' allows to insure tensorflow used a valid network
         name when the build is recalled."""

        self.__dict__.update(args)
        with self.graph.as_default():
            with tf.name_scope(self.name + "/"):
                self._build()
                self.sess.run(tf.initializers.global_variables())

    @abstractmethod
    def fit(self, *args):

        """The fit methods to train the neural network"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args):

        """ The predict method to make prediction"""
        raise NotImplementedError

    def save(self, path_folder: str) -> None:

        """
        Allows to save all the Tensorflow graph and all network parameters in a folder. The methods use the
        Tensorflow saver method and save all network parameters in a pickle file.

        Args
        ----

            path_folder : str
                Path of the folder where the network is saved.
        """

        assert path_folder.endswith("/")

        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path_folder + "archi.ckpt", global_step=0)

        write_pickle(self.get_params(), path_folder + "param.pkl")

    @abstractmethod
    def get_params(self) -> dict:

        """Return all network parameters."""

        return {
            "optimizer_name": self.optimizer_name,
            "learning_curve": self.learning_curve,
            'name': self.name}

    def _check_and_restore(self, path_folder):

        """Check if all folder and file exist and restore the graph and all class attributes then."""

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
                Path of the folder where the network is saved.
        """

        tf.reset_default_graph()
        self._check_and_restore(path_folder)

        env.RESTORE = True

        try:
            with self.graph.as_default():
                with tf.name_scope(self.name):
                    self._build()
        except Exception as e:
            raise TypeError(traceback.format_exc())
        finally:
            env.RESTORE = False

    def _get_optimizer(self):

        """Return the valid optimizer."""

        if self.optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optimizer_name == "SGD":
            return tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer_name == "Adam":
            return tf.train.AdamOptimizer(self.learning_rate)
        else:
            list_optimizer = ['RMSProp', 'SGD', 'Adam']
            raise TypeError(
                f"{self.optimizer_name} isn't a valid optimiser. optimiser_type must be in {list_optimizer}")

    def _minimize(self, f: tf.Tensor, name: str = "optimizer"):

        """
        Return an optimizer which minimize a tensor f. Add all paramters store in the UPDATE_OPS such that all moving
        mean and variance parameters of a batch normalization.

        Attributes:

            f : Tensor
                function to minimize

            name : str
                name of the tensor optimizer.

        Output
            Tensorflow optimizer
        """

        if env.RESTORE:
            return get_tf_operation(name, self.graph)
        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                return self._get_optimizer().minimize(f, name=name)

    def _placeholder(self, dtype: tf.DType, shape: (Tuple, None), name: str):

        """
        Set or restore a placeholder.

        Attributes:

        dtype : Tensorflow type
            Type of the placeholder

        shape : Tuple
            Size of the placeholder

        name : str
            name of the placeholder

        """

        if env.RESTORE:
            return get_tf_tensor(name, self.graph)
        else:
            is_not_in_graph(name, self.graph)
            return tf.placeholder(dtype, shape, name)
