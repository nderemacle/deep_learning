import sys

import numpy as np
import tensorflow as tf

from core.algo.MLP import MlpRegressor


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.power(y - y_pred, 2)))


def main():
    # Load and prepare data
    boston = tf.keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = boston.load_data()

    mu_y, sigma_y = y_train.mean(), y_train.std()
    mu_x, sigma_x = x_train.mean(0), x_train.std(0)

    x_train, x_test = (x_train - mu_x) / sigma_x, (x_test - mu_x) / sigma_x
    y_train = (y_train - mu_y) / sigma_y
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    # Build and train a first time the network
    reg = MlpRegressor(name="Boston_Regressor", use_gpu=False)

    reg.build(layer_size=(32, 64),
              input_dim=13,
              output_dim=1,
              act_funct='relu',
              keep_proba=1.,
              batch_norm=True,
              batch_renorm=True,
              law_name='uniform',
              law_param=1e-2,
              penalization_rate=1.,
              penalization_type="L2",
              optimizer_name="Adam",
              decay=0.99,
              decay_renorm=0.99,
              epsilon=0.001)

    print("First training begin:")
    reg.fit(x=x_train,
            y=y_train,
            n_epoch=50,
            batch_size=32,
            learning_rate=1e-3,
            rmin=1,
            rmax=1,
            dmax=0,
            verbose=True)

    # Make prediction
    y_train_predict = reg.predict(x=x_train, batch_size=32) * sigma_y + mu_y
    y_test_predict = reg.predict(x=x_test, batch_size=32) * sigma_y + mu_y
    print("Score train:  {}".format(rmse(y_train * sigma_y + mu_y, y_train_predict)))
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
            n_epoch=50,
            batch_size=32,
            learning_rate=1e-3,
            rmin=1 / 3,
            rmax=3,
            dmax=5,
            verbose=True)

    # Make prediction
    y_train_predict = reg.predict(x=x_train, batch_size=32) * sigma_y + mu_y
    y_test_predict = reg.predict(x=x_test, batch_size=32) * sigma_y + mu_y
    print("Score train:  {}".format(rmse(y_train * sigma_y + mu_y, y_train_predict)))
    print("Score test:  {}".format(rmse(y_test, y_test_predict)))

    # Save the network and free the memory
    reg.save(path_folder="output/MlpCLassifier/")
    del reg

    sys.exit(0)


if __name__ == '__main__':
    main()
