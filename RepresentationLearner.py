import keras
from keras import layers
import tensorflow as tf
from data_processing import data_augmentation, processer
class RepresentationLearner(keras.Model):
    def __init__(self, encoder, projection_unites, num_augmentations, temperature=1.0, dropout_rate=0.1, i2_normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

        self.projector = keras.Sequential([
            layers.Dropout(dropout_rate),
            layers.Dense(units=projection_unites, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.i2_normalize = i2_normalize
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vector, batch_size):
        num_augmentations = keras.ops.shape(feature_vector)[0] // batch_size
        if self.i2_normalize:
            feature_vector = keras.utils.normalize(feature_vector)

        logits = (
                tf.linalg.matmul(feature_vector, feature_vector, transpose_b=True) / self.temperature
        )

        logits_max = keras.ops.max(logits, axis=1)
        logits = logits - logits_max

        targets = keras.ops.tile(
            tf.eye(batch_size), [num_augmentations, num_augmentations],
        )

        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        preprocessed = processer(inputs)
        augmented = []

        for _ in range(self.num_augmentations):
            augmented_img = data_augmentation(preprocessed)
            augmented.append(self.encoder(augmented_img))

        features = layers.Concatenate(axis=0)(augmented)
        return self.projector(features)


    def train_step(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]

        with tf.GradientTape() as tape:
            features_vector = self(inputs, training=True)
            loss = self.compute_contrastive_loss(features_vector, batch_size)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        features_vector = self(inputs, training=False)
        loss = self.compute_contrastive_loss(features_vector, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
