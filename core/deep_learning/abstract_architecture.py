import os
import traceback
from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple
import numpy as np

import core.deep_learning.env as env
from core.utils.reader_writer import write_pickle, read_pickle
from core.deep_learning.tf_utils import get_tf_operation, get_tf_tensor

class AbstractArchitecture(ABC):

    """
    This abstract class defines the cortex for a deep learning architecture. It allows to well configure the
    tensorflow session and to launch the build or restore process. In addition it give access to functional methods
    such that optimizer setting.

    Attributes:

        name : str
            name of the network

        use_gpu: bool
            If true train the network on a single GPU otherwise used all cpu. Parallelism settign can be improve with
            future version

        graph : tf.Graph
            graph of the network usefull to well define new tensor and restore them.

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
    """

    def __init__(self, name : str, use_gpu : bool = False):

        self.name = name
        self.use_gpu = use_gpu
        self.graph = tf.Graph()
        self.sess = self._set_session()
        self.optimizer_name = "Adam"
        self.learning_curve = []
        self.learning_rate = None
        self.keep_proba_tensor = None

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

        self.learning_rate = self._placeholder(tf.float32, None, name=f"{self.name}/learning_rate")
        self.keep_proba_tensor = self._placeholder(tf.float32, None, name=f"{self.name}/keep_proba_tensor")

    def build(self, **args):

        """ Methods to build the Neural Network cortex. Te child methods can call the parent method to initialize
            all Variable."""

        self.__dict__.update(args)
        with self.graph.as_default():
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

    def save(self, path_folder : str):

        """This methods allow to save all the tensorflow graph in a folder. The methods use the tensorflow saver method
            and save all network parameters in a pickle."""

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

        with self.graph.as_default() as graph:
            saver = tf.train.import_meta_graph(path, clear_devices=True)
            saver.restore(self.sess, tf.train.latest_checkpoint(path_folder))

    def restore(self, path_folder : str):

        """Restore the graph by activating all restoration process. The RESTORE environment is first set to True
           to activate all class restoration methods when build is call. If a failure append, set the RESTORE
           environment variable to False and raise the error.


        Attribute:

            path_folder : str
                path of the folder where all network parameters are saved
        """

        tf.reset_default_graph()
        self._check_and_restore(path_folder)

        env.RESTORE = True

        try:
            with self.graph.as_default():
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

    def _minimize(self, f : tf.Tensor, name : str):

        """
        Return an optimizer which minimize a tensor f.

        Attribute:

            f : Tensor
                function to minimize

            name : str
                name of the tensor optimizer.

        Output
            Tensorflow optimizer
        """

        if env.RESTORE:
            return get_tf_operation(f"{self.name}/{name}", self.graph)
        else:
            return self._get_optimizer().minimize(f, name=f"{self.name}/{name}")

    def _placeholder(self, dtype: tf.DType ,   shape : Tuple , name : str):

        """
        Set or restore a placeholder.

        Attribute:

        dtype : Tensorflow type
            Type of the placeholder

        shape : Tuple
            Size of the placeholder

        name : str
            name of the placeholder

        """

        if env.RESTORE:
            return get_tf_tensor(name + ":0", self.graph)
        else:
            return tf.placeholder(dtype, shape, name)

    def _check_array(self, x : np.ndarray, shape : Tuple):

        assert isinstance(x, np.ndarray)
        assert np.all([x_s == s if s != -1 else True for x_s, s in zip(x.shape, shape)])
        assert np.all(np.isfinite(x))