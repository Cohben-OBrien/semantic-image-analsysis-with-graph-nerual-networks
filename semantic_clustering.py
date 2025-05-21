import keras
from keras import layers
from data_processing import processer, data_augmentation

class ClustersConsistencyLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, target, similarity, sample_weight=None):
        target = keras.ops.ones_like(similarity)

        loss = keras.losses.binary_crossentropy(
            y_true=target,
            y_pred=similarity,
            from_logits=True
        )

        return keras.ops.mean(loss)
#%%
class ClustersEntropyLoss(keras.losses.Loss):
    def __init__(self, entropy_loss_weight=1.0):
        super().__init__()
        self.entropy_loss_weight = entropy_loss_weight

    def __call__(self, target, cluster_probablities, similarity, sample_weight=None):
        num_clusters = keras.ops.cast(
            keras.ops.shape(cluster_probablities)[-1], 'float32'
        )

        target = keras.ops.log(num_clusters)
        cluster_probablities = keras.ops.mean(cluster_probablities, axis=0)
        cluster_probablities = keras.ops.clip(cluster_probablities, 1e-8, 1.0)

        entropy = -keras.ops.sum(
            cluster_probablities * keras.ops.log(cluster_probablities),
            )
        loss = target - entropy

        return loss
#%%
def create_clustering_model(encoder, num_clusters, input_shape, name=None):
    inputs = keras.Input(shape=input_shape)
    preprocessed = processer(inputs)
    augmented = data_augmentation(preprocessed)
    features = encoder(augmented)
    outputs = layers.Dense(units=num_clusters, activation='softmax')(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model
#%%
def create_clustering_learner(clustering_model, input_shape, kn=5):
    anchor = keras.Input(shape=input_shape, name='anchor')
    neighbours = keras.Input(shape=tuple([kn]) + input_shape, name='neighbours')

    neighbours_reshaped = keras.ops.reshape(neighbours, tuple([-1]) + input_shape)

    anchor_clustering = clustering_model(anchor)
    neighbour_clustering = clustering_model(neighbours_reshaped)
    neighbour_clustering = keras.ops.reshape(
        neighbour_clustering,
        (-1, kn, keras.ops.shape(neighbour_clustering)[-1])
    )

    simlarity = keras.ops.einsum(
        "bij,bkj->bik",
        keras.ops.expand_dims(anchor_clustering, axis=1),
        neighbour_clustering,
    )

    simlarity = layers.Lambda(
        lambda x: keras.ops.squeeze(x, axis=1), name='simlarity'
    ) (simlarity)

    model = keras.Model(
        inputs=[anchor, neighbours],
        outputs=[simlarity, anchor_clustering],
        name='clustering_learner'
    )

    return model