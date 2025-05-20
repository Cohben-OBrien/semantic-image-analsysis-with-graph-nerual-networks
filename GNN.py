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

class GraphConvLayer(layers.Layer):
    def __init__(self,
                 hidden_units,
                 dropout_rate=0.2,
                 aggregation='mean',
                 combination_type='concat',
                 normalization=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.aggregation = aggregation
        self.combination_type = combination_type
        self.normalization = normalization

        self.ffn_prepare = ffn(hidden_units, dropout_rate)
        if self.combination_type == 'gru':
            self.update_fn = create_gru(hidden_units, dropout_rate)
        else:
            self.update_fn = ffn(hidden_units, dropout_rate)


    def prepare(self, node_representation, weights=None):
        messages = self.ffn_prepare(node_representation)

        if weights is not None:
            messages = messages * tf.expand_dims(weights, axis=-1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_representation):
        num_nodes = node_representation.shape[0]
        print(self.aggregation)
        if self.aggregation == "sum":
            aggregated_messages = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation == "mean":
            aggregated_messages = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation == "max":
            aggregated_messages = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else: raise ValueError(f'Unknown aggregation type: {self.aggregation}')

        return aggregated_messages

    def update(self, node_repesemtation, aggregated_messages):

        if self.combination_type == 'gru':
            #create a sequence of two elements for the GRU layer
            h = tf.stack([node_repesemtation, aggregated_messages], axis=1)
        elif self.combination_type == 'concat':
            h = tf.concat([node_repesemtation, aggregated_messages], axis=1)
        elif self.combination_type == 'add':
            h = node_repesemtation + aggregated_messages
        else:
            raise ValueError(f'Unknown combination type: {self.combination_type}')

        node_embeding = self.update_fn(h)
        if self.combination_type == 'gru':
            node_embeding = tf.unstack(node_embeding, axis=1)[-1]

        if self.normalization:
            node_embeding = tf.nn.l2_normalize(node_embeding, axis=1)
        return node_embeding

    def call(self, inputs):
        node_repesentation, edges, edge_weights = inputs

        node_indices, neighbors_indices = edges[0], edges[1]

        neighbour_representation = tf.gather(node_repesentation, neighbors_indices)

        neighbour_massages = self.prepare(neighbour_representation, edge_weights)

        aggregated_messages = self.aggregate(
            node_indices, neighbour_massages, node_repesentation
        )

        return self.update(node_repesentation, aggregated_messages)



#%%
class GNNNodeClassification(tf.keras.Model):
    def __init__(self,
                 graph_info,
                 num_classes,
                 hidden_units,
                 aggregation_type="mean",
                 combination_type='concat',
                 dropout_rate=0.2,
                 normalize=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        node_features, edges, edge_weights = graph_info

        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights

        self.preprocess = ffn(hidden_units, dropout_rate, name=f'preprocess')

        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])

        self.edge_weights / tf.math.reduce_sum(self.edge_weights)


        self.conv1 = GraphConvLayer(hidden_units, dropout_rate,
                                    aggregation=aggregation_type,
                                    combination_type=combination_type,
                                    normalization=normalize,
                                    name='graph_conv1')

        self.conv2 = GraphConvLayer(hidden_units, dropout_rate,
                                    aggregation=aggregation_type,
                                    combination_type=combination_type,
                                    normalization=normalize,
                                    name='graph_conv2')
        self.postprocess = ffn(hidden_units, dropout_rate, name=f'postprocess')
        self.compute_logits = layers.Dense(num_classes, name='logits')



    def call(self, input_node_indices):
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x + x1

        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x + x2

        x = self.postprocess(x)

        print(input_node_indices)
        node_embedding = tf.gather(x, input_node_indices)

        return self.compute_logits(node_embedding)

