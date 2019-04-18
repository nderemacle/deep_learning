import sys

import numpy as np
import tensorflow as tf

from core.algo.ConvNet import ConvNetClassifier


def one_hot_encoding(x: np.ndarray):
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 1
    assert x.min() == 0

    n_values = np.max(x) + 1
    return np.eye(n_values)[x]


def main():
    # Load and prepare data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    # Build and train a first time the network
    clf = ConvNetClassifier(name="MNIST_Classifier", use_gpu=False)

    conv_params = [
        {"type": "CONV", "shape": (3, 3, 16), "stride": (2, 2), "padding": "VALID", "add_bias": True,
         "act_funct": None,
         "dilation": None}
    ]
    clf.build(conv_params=conv_params,
              fc_size=(128, 128),
              input_dim=(28, 28, 1),
              output_dim=10,
              act_funct='relu',
              keep_proba=0.8,
              batch_norm=True,
              batch_renorm=True,
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
            n_epoch=5,
            batch_size=32,
            learning_rate=1e-2,
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
    clf.save(path_folder="output/ConvNetClassifier/")
    del clf

    # Restore the mlp
    clf = ConvNetClassifier(use_gpu=False)
    clf.restore(path_folder="output/ConvNetClassifier/")

    # Continue the training
    print("Second training begin:")
    clf.fit(x=x_train,
            y=one_hot_encoding(y_train),
            n_epoch=5,
            batch_size=32,
            learning_rate=1e-3,
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
    clf.save(path_folder="output/ConvNetClassifier/")
    del clf

    sys.exit(0)


if __name__ == '__main__':
    main()
