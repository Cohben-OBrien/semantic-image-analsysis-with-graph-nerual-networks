import pandas as pd
import numpy as np
import keras
from keras import layers
import tensorflow as tf

class GraphAttention(layers.Layer):
    def __init__(
            self,
            units,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):


        # node states = feature vectors - edges = edge listy
        node_states, edges = inputs

        # Linearly transform node states
        # node_states (feature vector) * kernal
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        # match traformed node states with edges
        node_states_expanded = tf.gather(node_states_transformed, edges)

        # Concatenate the transformed node states of source and target nodes
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )

        # Compute attention scores by applying a linear transformation followed by a leaky ReLU activation
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )

        # Reshape to remove the last dimension
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        # Clip the attention scores to avoid overflow in exp
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))

        # Sum attention scores for each source node
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )

        # Normalize attention scores by dividing by the sum of attention scores for each source node
        segment_ids = tf.cast(edges[:, 0], tf.int32)
        # Expand attention_scores_sum to match the shape of attention_scores
        expanded_attention_scores_sum = tf.gather(attention_scores_sum, segment_ids)
        # Avoid division by zero
        attention_scores_norm = attention_scores / expanded_attention_scores_sum


        # (3) Gather node states of neighbors, apply attention scores and aggregate
        # Gather the transformed node states of neighbors using the edges
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        # Multiply the gathered node states by the normalized attention scores
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        ) # output shape: (num_nodes, units)


        return out



#%%
class MultiHeadGraphAttention(layers.Layer):
    def __init__(
            self,
            units,
            num_heads,
            merge_type='concat',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        features, indices = inputs # features: node states, indices: edges
        outputs = [
            attention_layer([features, indices]) for attention_layer in self.attention_layers
        ] # outputs is a list of tensors, each with shape (num_nodes, units)

        # Merge the outputs of the attention heads

        if self.merge_type == 'concat':
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

        return tf.nn.relu(outputs) #return the merged outputs with ReLU activation


class GraphAttentionNetwork(keras.Model):
    def __init__(
            self,
            node_states,
            edges,
            hidden_units,
            num_heads,
            num_layers,
            output_dim,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation='relu') # preprocess the node states before passing to attention layers
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers) # create multiple attention layers creating multiple heads
        ]
        self.output_layer = layers.Dense(output_dim)  # output layer to produce final output as shape (num_nodes, output_dim)

    def call(self, inputs):
        print(inputs)
        node_states, edges = inputs # node_states: feature vectors, edges: edge list as a tensor of shape (num_edges, 2)
        x = self.preprocess(node_states) # x shape: (num_nodes, hidden_units * num_heads
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x # residual connection
        output = self.output_layer(x) # output shape: (num_nodes, output_dim)
        return output

    def train_step(self, data):
        indices, labels = data # geting indices and labels from the data defined in the fit method as x=edges, y=labels

        with tf.GradientTape() as tape: # record the gradients for backpropagation
            outputs = self([self.node_states, self.edges]) # forward pass through the model
            loss = self.compiled_loss(labels, tf.gather(outputs, indices)) # compute the loss using the gathered outputs and labels

        grads = tape.gradient(loss, self.trainable_weights) # compute the gradients of the loss with respect to the trainable weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) # apply the gradients to the trainable weights
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices)) # update the metrics using the gathered outputs and labels

        return {m.name: m.result() for m in self.metrics} # return the metrics as a dictionary with metric names as keys and their values as values

    def predict_step(self, data):
        indices, labels = data
        output = self([self.node_states, self.edges]) # forward pass through the model
        probs = tf.nn.softmax(output, axis=-1) # apply softmax to the output to get probabilities
        return tf.gather(probs, indices) # gather the probabilities using the indices from the data

    def test_step(self, data):
        indices, labels = data
        outputs = self([self.node_states, self.edges])
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))

        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))

        return {m.name: m.result() for m in self.metrics}

