import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input


def resenet_50(x_img: np.ndarray):
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
                      pooling=None, classes=1000)

    return resnet.predict(preprocess_input(x_img))
