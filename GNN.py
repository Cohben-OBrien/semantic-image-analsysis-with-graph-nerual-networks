import keras
from keras import layers
def ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation='relu'))

    return keras.Sequential(fnn_layers, name=name)


def create_baseline_model(hidden_units, num_classes, dropout_rate):
    inputs = keras.Input(shape=(32, 32, 3))
    x = ffn(hidden_units, dropout_rate, name='ffn_block1')(inputs)

    for block_idx in range(4):
        x1 = ffn(hidden_units, dropout_rate, name=f'ffn_block{block_idx+2}')(x)
        x = layers.Add(name=f'skip connection {block_idx + 2}')([x, x1])

    logits = layers.Dense(num_classes, name='logits')(x)

    return keras.Model(inputs=inputs, outputs=logits, name='baseline_model')

