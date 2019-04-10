from typing import Sequence, Union, Callable

import numpy as np
import tensorflow as tf

from core.utils.validation import is_in_graph, is_not_in_graph


def random_law(shape: Sequence[int], law_name: str, law_param: float, dtype: tf.DType = tf.float32) -> tf.Tensor:
    """
    Return a random generator tensor.

    Args
    ----

        shape: Sequence[int]
            Dimension of the tensor generated.

        law_name: str
            Law of the random law to use. Must be "normal" for normal law or "uniform" for uniform law.

        law_param: float
            Law parameters which is dependent to the law selected. If uniform, all tensor
            elements are initialized using a U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        dtype: tf.DType
            Type for generated tensor.

    Returns
    -------

        tf.Tensor
            Random generator tensor.

    """

    if law_name == "normal":
        return tf.random_normal(shape=shape, stddev=law_param, dtype=dtype)
    elif law_name == "uniform":
        return tf.random_uniform(shape=shape, minval=-law_param, maxval=law_param, dtype=dtype)
    else:
        list_law = ['normal', 'uniform']
        raise TypeError(f"{law_name} isn't a valide law_name. Name must be in {list_law}")

def variable(shape: Sequence[int], initial_value: Union[np.ndarray, None] = None, law_name: str = "uniform",
             law_param: float = 0.1, name: Union[str, None] = None, dtype: tf.DType = tf.float32) -> tf.Variable:
    """
    Return a Tensorflow Variable object uninitialised using either a random number law generator or a deterministic
    value allowing the use no implemented random law generator or the use of transfer learning methods.

    Args
    ----

        shape: Sequence[int]
            Dimension of the tensor generated.

        initial_value: np.array
            Optional initial value for the variable allowing the use of not implemented initialization methods.

        law_name: str
            Law of the random law to use. Must be "normal" for normal law or "uniform" for uniform law.

        law_param: float
            Law parameters which is dependent to the law selected. If uniform, all tensor
            elements are initialized using a U(-law_params, law_params) and if normal all parameters are initialized
            using a N(0, law_parameters).

        name: str
            Tensor name.

        dtype: Tensorflow type
            Type of values generated.

    Returns
    -------

        tf.Variable
            Tensorflow Variable object.
    """

    if initial_value is None:
        initial_value = random_law(shape, law_name, law_param, dtype)
    else:
        assert initial_value.shape == shape

    is_not_in_graph(name)

    return tf.Variable(initial_value, name=name)


def get_act_funct(name: str = 'relu') -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Return a Tensor activation method.

    Args
    ----

        name: str
            Name of the activation function. Must be included in: ['relu', 'sigmoid', 'tanh', 'relu6',
            'crelu', 'elu', 'softplus', 'softsign']

    Returns
    -------

       Activation function tensor.

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


def get_tf_tensor(name: str, graph: Union[tf.Graph, None] = None) -> tf.Tensor:
    """
    Return the Tensorflow tensor with the given name. Each tensor must be include inside a scope or a sub scope.
    Regarding the mechanise of the framework, if a tensor is instance at the network level its name should be
    "NetworkName/TensorName" or should be "NetworkName/OperatorName/TensorName" if instance in an operator object.

    Args
    ----

        name: str
            Name of the tensor which must be included in the graph.

        graph: tf.Graph, None
            Tensorflow grah object. If None take the default graph.

    Returns
    -------

        tf.Tensor
            Restored tensor link to the entry name.
    """

    _graph = tf.get_default_graph() if graph is None else graph

    with _graph.as_default() as g:
        tensor_name = g.get_name_scope() + "/" + name
        is_in_graph(tensor_name, g)
        return g.get_tensor_by_name(tensor_name + ":0")


def get_tf_operation(name: str, graph: Union[tf.Graph, None] = None) -> tf.Tensor:
    """
    Return the Tensorflow operation with the given name. Each operation must be include inside a scope or a sub scope.
    Regarding the mechanise of the framework, if an operation is instance at the network level its name should be
    "NetworkName/OperationName" or should be "NetworkName/OperatorName/OperationName" if instance in an operator object.

    Args
    ----

        name: str
            Name of the tensor which must be included in the graph.

        graph : tf.Graph, None
            Tensorflow grah object. If None take the default graph.

    Returns
    -------

        tf.Tensor
            Restored tensor link to the entry name.

    """

    _graph = tf.get_default_graph() if graph is None else graph
    with _graph.as_default() as g:
        operation_name = g.get_name_scope() + "/" + name
        is_in_graph(operation_name, g)
        return g.get_operation_by_name(operation_name)


def identity(x: tf.Tensor, name: str, graph: Union[tf.Graph, None] = None) -> tf.Tensor:
    """
    Give or force a name for a given tensor.

    Args
    ----

        x: tf.Tensor
            Tensor to name.

        name:  str
            Name for the tensor.

        graph: tf.Graph, None
            Tensorflow grah object. If None take the default graph.

    Returns
    -------

        Renamed tensor.

    """
    _graph = tf.get_default_graph() if graph is None else graph

    is_not_in_graph(name, _graph)

    return tf.identity(x, name)


def get_optimizer(name: str, learning_rate: Union[tf.Tensor, float]) -> tf.train.Optimizer:

    """
    Return a Tensorflow optimizer Tensor.

    Args
    ----
        learning_rate: tf.Tensor, float
            Learning rate to use for optimization.

        name: str
            Name of the optimizer. Must be RMSProp, SGD or Adam.

    Returns
    -------

        tf.train.Optimizer
            Tensorflow optimizer object.
    """

    if name == "RMSProp":
        return tf.train.RMSPropOptimizer(learning_rate)
    elif name == "SGD":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif name == "Adam":
        return tf.train.AdamOptimizer(learning_rate)
    else:
        list_optimizer = ['RMSProp', 'SGD', 'Adam']
        raise TypeError(
            f"{name} isn't a valid optimiser. optimiser_type must be in {list_optimizer}")