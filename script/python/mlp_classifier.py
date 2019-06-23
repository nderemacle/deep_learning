import sys

import numpy as np
import tensorflow as tf

from core.algo.MLP import MlpClassifier


def one_hot_encoding(x: np.ndarray):
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    assert x.min() == 0

    n_values = np.max(x) + 1
    return np.eye(n_values)[x]


def relu(x: tf.Tensor) -> tf.Tensor:
    return tf.maximum(x, 0)


def main():
    # Load and prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0

    # Build and train a first time the network
    clf = MlpClassifier(name="MNIST_Classifier", use_gpu=False)

    clf.build(layer_size=(64, 128),
              input_dim=784,
              output_dim=10,
              act_funct=relu,
              dropout=False,
              batch_norm=False,
              batch_renorm=False,
              law_name='uniform',
              law_param=1e-2,
              penalization_rate=0.,
              penalization_type='L2',
              optimizer_name="Adam",
              decay=0.99,
              decay_renorm=0.99,
              epsilon=0.001)

    print("First training begin:")
    clf.fit(x=x_train,
            y=one_hot_encoding(y_train),
            n_epoch=2,
            batch_size=32,
            learning_rate=1e-3,
            keep_proba=0.8,
            rmin=1,
            rmax=1,
            dmax=0,
            verbose=True)

    # Make prediction
    y_train_predict = clf.predict(x=x_train, batch_size=32)
    y_test_predict = clf.predict(x=x_test, batch_size=32)
    print("Score train:  {}".format((y_train == y_train_predict).mean()))
    print("Score test:  {}".format((y_test == y_test_predict).mean()))

    # Save the network and free the memory
    clf.save(path_folder="output/MlpClassifier/")
    del clf

    # Restore the mlp
    clf = MlpClassifier(use_gpu=False)
    clf.restore(path_folder="output/MlpClassifier/")

    # Continue the training
    print("Second training begin:")
    clf.fit(x=x_train,
            y=one_hot_encoding(y_train),
            n_epoch=10,
            batch_size=32,
            learning_rate=1e-3,
            keep_proba=0.8,
            rmin=1 / 3,
            rmax=3,
            dmax=5,
            verbose=True)

    # Make prediction
    y_train_predict = clf.predict(x=x_train, batch_size=32)
    y_test_predict = clf.predict(x=x_test, batch_size=32)
    print("Score train:  {}".format((y_train == y_train_predict).mean()))
    print("Score test:  {}".format((y_test == y_test_predict).mean()))

    # Save the network and free the memory
    clf.save(path_folder="output/MlpClassifier/")
    del clf

    sys.exit(0)


if __name__ == '__main__':
    main()
