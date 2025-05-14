import keras

from keras import layers



target_size = 32

processer =  keras.Sequential([
    layers.Resizing(target_size, target_size),
    layers.Normalization(),
])


data_augmentation = keras.Sequential([
    layers.RandomTranslation(
        height_factor=(-0.2, 0.2),
        width_factor=(-0.2, 0.2),
        fill_mode='nearest',
    ),
    layers.RandomFlip(
        mode='horizontal',
    ),
    layers.RandomRotation(
        factor=0.15, fill_mode='nearest',
    ),
    layers.RandomZoom(
        height_factor=(-0.3, 0.1),
        width_factor=(-0.3, 0.1),
        fill_mode='nearest',
    )

])