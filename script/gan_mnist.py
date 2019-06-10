import sys

import numpy as np
import tensorflow as tf

from core.algo.GAN import Gan


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

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    del x_train
    del x_test
    del y_test
    del y_train
    x = x.reshape(-1, 784) / 255.0

    # Build and train a first time the network
    gen = Gan(name="GAN_MNIST", use_gpu=False)

    gen.build(G_layer_size=(256, 512),
              D_layer_size=(512, 256),
              input_dim=784,
              noise_dim=100,
              D_act_funct='leaky_relu',
              G_act_funct='leaky_relu',
              D_final_funct='sigmoid',
              G_final_funct='sigmoid',
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
    gen.fit(x=x,
            n_epoch=10,
            batch_size=100,
            learning_rate=2e-4,
            keep_proba=0.3,
            verbose=True)

    # Save the network and free the memory
    gen.save(path_folder="output/GanMnist/")
    del gen

    # Restore the mlp
    gen = Gan(name="GAN_MNIST", use_gpu=False)
    gen.restore(path_folder="output/GanMnist/")

    # Continue the training
    print("Second training begin:")
    gen.fit(x=x,
            n_epoch=2,
            batch_size=256,
            learning_rate=1e-4,
            verbose=True)

    # Save the network and free the memory
    gen.save(path_folder="output/GanMnist/")
    del gen

    sys.exit(0)


if __name__ == '__main__':
    main()
