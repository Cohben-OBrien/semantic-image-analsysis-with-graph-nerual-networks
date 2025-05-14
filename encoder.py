import keras
from keras import layers


def create_encoder(Representation_dims):
    return keras.Sequential([
        keras.applications.ResNet50V2(
            include_top=False,
            weights=None,
            pooling='avg'
        ),
        layers.Dense(Representation_dims)
    ])