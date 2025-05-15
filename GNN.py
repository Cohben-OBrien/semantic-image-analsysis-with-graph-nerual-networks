import keras
from keras import layers
import tensorflow as tf
from networkx.classes import neighbors


def ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation='relu'))

    return keras.Sequential(fnn_layers, name=name)


def create_baseline_model(hidden_units, num_classes, dropout_rate):
    num_features = hidden_units[0]
    inputs = keras.Input(shape=(num_features, ), name='input_features')
    x = ffn(hidden_units, dropout_rate, name='ffn_block1')(inputs)

    for block_idx in range(4):
        x1 = ffn(hidden_units, dropout_rate, name=f'ffn_block{block_idx+2}')(x)
        x = layers.Add(name=f'skip_connection{block_idx + 2}')([x, x1])

    logits = layers.Dense(num_classes, name='logits')(x)

    return keras.Model(inputs=inputs, outputs=logits, name='baseline_model')

def create_gru(hidden_units, dropout_rate):
    inputs = keras.layers.Input(shape=(2, hidden_units[0]))
    x = inputs

    for units in hidden_units:
        x = layers.GRU(
            units,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=True,
            droupout=dropout_rate,
            return_state=False,
            recurrent_dropout=dropout_rate
        )(x)

    return keras.Model(inputs=inputs, outputs=x)

class GrahnConvLayer(layers.Layer):
    def __init__(self,
                 hidden_units,
                 dropout_rate=0.2,
                 aggregation_type='mean',
                 combination_type='concat',
                 normalize=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.fnn_prepare = ffn(hidden_units, dropout_rate)

        if self.combination_type == 'gru':
            self.update_fm = create_gru(hidden_units, dropout_rate)
        else:
            self.update_fm = ffn(hidden_units, dropout_rate)

        def prepare(self, node_rep, weights=None):
            mesages = self.fnn_prepare(node_rep)
            if weights is not None:
                messages = self.fnn_prepare * tf.expand_dims(weights, -1)
            return messages

        def aggregate(self, node_indices, neigbour_messages, node_reps):
            num_nodes = node_reps.shape[0]

            if self.aggregation_type == 'sum':
                aggregated_messages = tf.math.unsorted_segment_sum(
                    neigbour_messages, node_indices, num_segments=num_nodes
                )
            elif self.aggregation_type == 'mean':
                aggregation_messages = tf.math.unsorted_segment_mean(
                    neigbour_messages, node_indices, num_segments=num_nodes
                )
            elif self.aggregation_type == 'max':
                aggregation_messages = tf.math.unsorted_segment_max(
                    neigbour_messages, node_indices, num_segments=num_nodes
                )
            else:
                raise ValueError(f'invalid aggregation type: {self.aggregation_type}')

        def update(self, node_rep, aggregated_messages):
            if self.combination_type == 'gru':
                h = tf.stack([node_rep, aggregated_messages], axis=1)
            elif self.combination_type == 'concat':
                h = tf.concat([node_rep, aggregated_messages], axis=1)
            elif combination_type == 'add':
                h = node_rep + aggregated_messages
            else:
                raise ValueError(f'invalid combination type: {self.combination_type}')

            node_embedding = self.update_fnn(h)
            if self.combination_type == 'gru':
                node_embedding = tf.unstack(node_embedding, axis=1)[-1]

            if self.normalize:
                node_embedding = tf.nn.l2_normalize(node_embedding, axis=-1)

            return node_embedding

        def call(self, inputs):
            node_reps, edges, edge_weights = inputs

            node_indices, neighbors_indices = edges[0], edges[1]

            neighbour_reps = tf.gather(node_reps, neighbors_indices)

            neighbour_messages = self.prepare(node_reps, edge_weights)
            aggreated_messages = self.aggregate(node_indices, neighbour_messages, node_reps)

            return self.update(node_reps, aggreated_messages)

