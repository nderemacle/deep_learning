from typing import Tuple

import numpy as np
import tensorflow as tf

from core.utils.validation import is_in_graph, is_not_in_graph


def random_law(shape: Tuple, law_name: str, law_param: float, dtype: tf.DType = tf.float32):
    """
    Return a Tensorflow random generator.

    Attributes:

        shape : Tuple
            Dimension of the tensor generate

        law_name : str
            Law of the random law to used. Must be "normal" for normal law or "uniform" for uniform law.

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
        return tf.random_uniform(shape=shape, minval=-law_param, maxval=law_param, dtype=dtype)
    else:
        list_law = ['normal', 'uniform']
        raise TypeError(f"{law_name} isn't a valide law_name. Name must be in {list_law}")


def variable(shape: Tuple, initial_value: np.ndarray = None, law_name: str = "uniform", law_param: float = 0.1,
             name: str = None, dtype: tf.DType = tf.float32):
    """
    Build a Tensorflow variable uninitialised using either a random number law generator or deterministic value
    allowing no implemented random law generator or the use of transfer learning methods.

    Attributes:

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

    dtype : Tensorflow type
        type of values generate

    Output:

        Tensorflow Variable object
    """

    if initial_value is None:
        initial_value = random_law(shape, law_name, law_param, dtype)
    else:
        assert initial_value.shape == shape

    is_not_in_graph(name)

    return tf.Variable(initial_value, name=name)


def get_act_funct(name: str = 'relu'):
    """
    Return a Tensorflow activation method.

    Attributes:

        name : str
            name of the activation function

    Output:

        activation function object
    """

    if name == 'relu':
        return tf.nn.relu
    elif name == 'sigmoid':
        return tf.nn.sigmoid
    elif name == 'tanh':
        return tf.nn.tanh
    elif name == 'relu6':
        return tf.nn.relu6
    elif name == 'crelu':
        return tf.nn.crelu
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'softsign':
        return tf.nn.softsign
    else:
        list_act_funct = ['relu', 'sigmoid', 'tanh', 'relu6', 'crelu', 'elu', 'softplus', 'softsign']
        raise TypeError(
            f"{name} isn't a valide activation function. Methods must be in {list_act_funct}")


def get_tf_tensor(name: str, graph: tf.Graph = None):
    """
    Return the Tensorflow tensor with the given name. Each tensor must be include a scope or a sub scope.
    Regarding the mechanise of the framework, if a tensor is instance at the network level its name should be
    "NetworkName/TensorName" or should be "NetworkName/OperatorName/TensorName" if instance in an operator object.

    Attributes:

        name : str
            Name of the tensor which must be include in the graph.

        graph : tf.Graph or None
            Tensorflow grah object. If None take the default graph.

    Output:
        Tensor
    """

    _graph = tf.get_default_graph() if graph is None else graph

    with _graph.as_default() as g:
        tensor_name = g.get_name_scope() + "/" + name
        is_in_graph(tensor_name, g)
        return g.get_tensor_by_name(tensor_name + ":0")


def get_tf_operation(name: str, graph: tf.Graph = None):
    """
    Return the Tensorflow operation with the given name. Each operation must be include a scope or a sub scope.
    Regarding the mechanise of the framework, if an operation is instance at the network level its name should be
    "NetworkName/OperationName" or should be "NetworkName/OperatorName/OperationName" if instance in an operator object.

    Attributes:

    name : str
        Name of the tensor which must be include in the graph.

    graph : tf.Graph or None
        Tensorflow grah object. If None take the default graph.

    Output:
        Tensor
    """

    _graph = tf.get_default_graph() if graph is None else graph
    with _graph.as_default() as g:
        operation_name = g.get_name_scope() + "/" + name
        is_in_graph(operation_name, g)
        return g.get_operation_by_name(operation_name)


def identity(x: tf.Tensor, name: str, graph: tf.Graph = None):
    _graph = tf.get_default_graph() if graph is None else graph

    is_not_in_graph(name, _graph)

    return tf.identity(x, name)
