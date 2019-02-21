import tensorflow as tf
from typing import Tuple
import numpy as np


def random_law(shape: Tuple, law_name: str, law_param: float, dtype: tf.DType = tf.float32):

    """
    Return a tensorflow random generator.

    Attribute:

        shape : Tuple
            Dimension of the tensor generate

        law_name : str
            Law of the ransom law to used. Must be "normal" for normal law or "uniform" for uniform law.

        law_params : float
            Law parameters which is dependent to the initialised law choose. If uniform, all tensor
            elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        dtype : Tensorflow type
            type of values generate

    Output:

        Tensorflow random generator

    """

    if law_name == "normal":
        return tf.random_normal(shape=shape, stddev=law_param, dtype=dtype)
    elif law_name == "uniform":
        return tf.random_uniform(shape=shape, minval=-law_param, maxval=law_param ,dtype=dtype)
    else:
        list_law = ['normal', 'uniform']
        raise TypeError(f"{law_name} isn't a valide law_name. Name must be in {list_law}")


def build_variable(shape: Tuple, initial_value: np.ndarray = None, law_name: str = "uniform", law_param: float = 0.1,
                   name: str= None, dtype: tf.DType = tf.float32):

    """
    Build a tensorflow variable uninitialised using either a random number law generator or deterministic value
    allowing no implemented random law generator or the use of transfer learning methods.

    Attribute:

    shape : Tuple
        Dimension of the tensor generate

    initial_value : array
        Optional initial value for the variable allowing to used not implemented initialized methods

    law_name : str
        Law of the ransom law to used. Must be "normal" for normal law or "uniform" for uniform law.

    law_params : float
        Law parameters which is dependent to the initialised law choose. If uniform, all tensor
        elements are initialized using U(-law_params, law_params) and if normal all parameters are initialized
        using a N(0, law_parameters).

    name : str
        tensor name

    dtype : tensorflow type
        type of values generate

    Output:

        Tensorflow Variable object
    """

    if initial_value is None:
        initial_value = random_law(shape, law_name, law_param, dtype)
    else:
        assert initial_value.shape == shape

    return tf.Variable(initial_value, name=name)


def get_act_funct(act_funct_type : str = 'relu'):

    """
    Return a tensorflow activation method

    Attribute:

        act_funct_type : str
            name of the activation function

    Output:

        activation function object
    """

    if act_funct_type == 'relu':
        return tf.nn.relu
    elif act_funct_type == 'sigmoid':
        return tf.nn.sigmoid
    elif act_funct_type == 'tanh':
        return tf.nn.tanh
    else:
        list_act_funct = ['relu', 'sigmoid', 'tanh']
        raise TypeError(
            f"{act_funct_type} isn't a valide activation function. Methods must be in {list_act_funct}")


def get_tf_tensor(name : str, graph : tf.Graph = None):

    """
    Return the tensorflow tensor with the given name

    Attribute:

        name : str
            Name of the tensor which must be include in the graph.

        graph : tf.Graph
            Tensorflow grah object

    Output:
        Tensor
    """

    _graph = tf.get_default_graph() if graph is None else graph

    with _graph.as_default() as g:

        return g.get_tensor_by_name(name)

def get_tf_operation(name : str, graph: tf.Graph = None):

    """
    Return the tensorflow operation with the given name


    Attribute:

    name : str
        Name of the tensor which must be include in the graph.

    graph : tf.Graph
        Tensorflow grah object

    Output:
        Tensor
    """

    _graph = tf.get_default_graph() if graph is None else graph
    with _graph.as_default() as g:
        return g.get_operation_by_name(name)

def get_all_tensor_name(graph: tf.Graph = None):
    """
    Return a list with all instance tensor in a graph.

    Attribute:

    graph : tf.Graph
        Tensorflow grah object

    Output:
        List of tensor name

    """

    _graph = tf.get_default_graph() if graph is None else graph

    return [t.name for t in _graph.get_operations()]