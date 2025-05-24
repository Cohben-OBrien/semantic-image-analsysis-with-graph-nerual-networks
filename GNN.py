from multiprocessing.util import sub_warning

import keras
from keras import layers
import tensorflow as tf
from networkx.classes import neighbors

def ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation ='gelu'))

    return keras.Sequential(fnn_layers,
                            name=name)

def create_gru(hidden_units, dropout_rate):
    inputs = keras.layers.Input(shape=(2, hidden_units[0]))
    x = inputs


    for units in hidden_units:
        x = layers.GRU(
            units=units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=True,
            dropout=dropout_rate,
            return_state=False,
            recurrent_dropout=dropout_rate,

        )(x)

    return keras.Model(inputs=inputs, outputs=x)


class GraphConvolution(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)

        self.output_dim = output_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                     shape=(input_shape[0][-1], self.output_dim),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, inputs):
        features, adjacency = inputs
        output = tf.matmul(adjacency, features)
        output = tf.matmul(output, self.kernel)
        return output



def create_gnn_encoder(feature_dim, num_nodes, num_classes,):
    feature_input = keras.Input(shape=(feature_dim,))
    adjacency_input = keras.Input(shape=(num_nodes,))

    x = GraphConvolution(64)([feature_input, adjacency_input])
    x = layers.ReLU()(x)
    x = GraphConvolution(32)([x, adjacency_input])
    x = layers.ReLU()(x)

    output = layers.Dense(num_classes, activation='softmax')(x)

    encoder = keras.Model(inputs=[feature_input, adjacency_input], outputs=output)

    return encoder



def create_projector(projection_dim=64, out_dim=64):
    return keras.Sequential([
        layers.Dense(out_dim, activation='relu'),
        layers.Dense(out_dim),
    ], name='projection')


class SimSiamGNN(keras.Model):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @staticmethod
    def negative_cosine_similarity(A, B):
        a = tf.math.l2_normalize(A, axis=-1)
        b = tf.math.l2_normalize(B, axis=-1)
        return -tf.reduce_mean(tf.reduce_sum(a * b, axis=-1))

    def train_step(self, data):
        (features1, adj1), (features2, adj2) = data

        with tf.GradientTape() as tape:
            z1 = self.encoder([features1, adj1])
            z2 = self.encoder([features2, adj2])

            p1 = self.projector(z1)
            p2 = self.projector(z2)

            loss = (self.negative_cosine_similarity(p1, tf.stop_gradient(z2)) +
                    self.negative_cosine_similarity(p2, tf.stop_gradient(z1)) *
                    0.5)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


class GraphDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 features,
                 adjacency,
                 batch_size=1,
                 augmented_fn=None):
        self.features = features
        self.adjacency = adjacency
        self.batch_size = batch_size
        self.augmented_fn = augmented_fn


    def __len__(self):
        return len(self.features) // self.batch_size

    def __getitem__(self, idx):
        feat1, adj1 = self.augmented_fn(self.features, self.adjacency)
        feat2, adj2 = self.augmented_fn(self.features, self.adjacency)

        return ([feat1, adj1], [feat2, adj2])


