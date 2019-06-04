from typing import Tuple, Union, Optional

import numpy as np
import tensorflow as tf


def check_array(x: np.ndarray, shape: Tuple):
    """
    Check is an object is a np.array with a valide shape and with only finite value.

    Attributes:

        x: np.array
            An object to apply array test.

        shape: Tuple
            Expected array dimension.
    """

    assert isinstance(x, np.ndarray)
    assert np.all([x_s == s if s != -1 else True for x_s, s in zip(x.shape, shape)])
    assert np.all(np.isfinite(x))


def get_all_tensor_name(graph: Optional[tf.Graph] = None):
    """
    Return a list with all instance tensor in a graph.

    Attributes:

    graph : tf.Graph or None
        Tensorflow grah object. If None take the default graph.

    Output:
        List of tensor name

    """

    _graph = tf.get_default_graph() if graph is None else graph

    return [t.name for t in _graph.get_operations() + tf.global_variables()]


def is_in_graph(name: str, graph: Optional[tf.Graph] = None):
    """
    Return a list with all instance tensor in a graph.

    Attributes:

    graph : tf.Graph or None
        Tensorflow grah object. If None take the default graph.

    name: str
        List of tensor name

    """

    _graph = tf.get_default_graph() if graph is None else graph

    if name not in get_all_tensor_name(graph=_graph):
        raise NameError(f"{name} does not exist.")


def is_not_in_graph(name: str, graph: Optional[tf.Graph] = None):
    """
    Return a list with all instance tensor in a graph.

    Attributes:

    graph : tf.Graph or None
        Tensorflow grah object. If None take the default graph.

    name: str
        List of tensor name

    """

    _graph = tf.get_default_graph() if graph is None else graph

    if name + ":0" in get_all_tensor_name(graph=_graph):
        raise NameError(f"{name} already exist.")


def check_tensor(x: tf.Tensor, shape: Optional[Tuple] = None, shape_dim: Optional[int] = None):
    if not isinstance(x, tf.Tensor):
        raise TypeError(f"Expected tf.Tensor type object but received {type(x)}.")

    if shape is not None:
        x_shape = tuple(s.value for s in x.shape)
        if x_shape != shape:
            raise TypeError(f"Expected tensor with shape {shape} but received tensor with shape {x_shape}.")

    if shape_dim is not None:
        if len(x.shape) != shape_dim:
            raise TypeError(f"Expected tensor with shape dimension {shape_dim} but received"
                            f" a tensor with shape_dim {x.shape.ndims}")


def check_variable(x: tf.Variable, shape: Optional[Tuple] = None, shape_dim: Optional[int] = None):
    if not isinstance(x, tf.Variable):
        raise TypeError(f"Expected tf.Variable type object but received {type(x)}.")

    if shape is not None:
        x_shape = tuple(s.value for s in x.shape)
        if x_shape != shape:
            raise TypeError(f"Expected variable with shape {shape} but received tensor with shape {x_shape}.")

    if shape_dim is not None:
        if len(x.shape) != shape_dim:
            raise TypeError(f"Expected variable with shame dimension {shape_dim} but received"
                            f" a tensor with shape_dim {x.shape.ndims}")
