import sys

import numpy as np
import tensorflow as tf

from core.algo.MLP import MlpRegressor


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.power(y - y_pred, 2)))


def sigmoid(x: tf.Tensor) -> tf.Tensor:
    return tf.sigmoid(x)


def main():
    # Load and prepare data
    boston = tf.keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = boston.load_data()

    y_max = y_train.max()
    mu_x, sigma_x = x_train.mean(0), x_train.std(0)
    x_train, x_test = (x_train - mu_x) / sigma_x, (x_test - mu_x) / sigma_x
    y_train /= y_max
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    # Build and train a first time the network
    reg = MlpRegressor(name="Boston_Regressor", use_gpu=False)

    reg.build(layer_size=(16,),
              input_dim=13,
              output_dim=1,
              act_funct='tanh',
              final_funct=sigmoid,
              dropout=False,
              batch_norm=True,
              batch_renorm=True,
              law_name='normal',
              law_param=1e-2,
              penalization_rate=0.,
              penalization_type="L2",
              optimizer_name="Adam",
              decay=0.99,
              decay_renorm=0.99,
              epsilon=0.01)

    print("First training begin:")
    reg.fit(x=x_train,
            y=y_train,
            n_epoch=100,
            batch_size=32,
            learning_rate=1e-2,
            keep_proba=0.8,
            rmin=1,
            rmax=1,
            dmax=0,
            verbose=True)

    # Make prediction
    y_train_predict = reg.predict(x=x_train, batch_size=32) * y_max
    y_test_predict = reg.predict(x=x_test, batch_size=32) * y_max
    print("Score train:  {}".format(rmse(y_train * y_max, y_train_predict)))
    print("Score test:  {}".format(rmse(y_test, y_test_predict)))

    # Save the network and free the memory
    reg.save(path_folder="output/MlpRegressor/")
    del reg

    # Restore the mlp
    reg = MlpRegressor(use_gpu=False)
    reg.restore(path_folder="output/MlpRegressor/")

    # Continue the training
    print("Second training begin:")
    reg.fit(x=x_train,
            y=y_train,
            n_epoch=100,
            batch_size=32,
            learning_rate=1e-3,
            keep_proba=0.8,
            rmin=0.33,
            rmax=3,
            dmax=5,
            verbose=True)

    # Make prediction
    y_train_predict = reg.predict(x=x_train, batch_size=32) * y_max
    y_test_predict = reg.predict(x=x_test, batch_size=32) * y_max
    print("Score train:  {}".format(rmse(y_train * y_max, y_train_predict)))
    print("Score test:  {}".format(rmse(y_test, y_test_predict)))

    # Save the network and free the memory
    reg.save(path_folder="output/MlpRegressor/")
    del reg

    sys.exit(0)


if __name__ == '__main__':
    main()
