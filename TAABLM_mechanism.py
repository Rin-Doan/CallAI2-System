import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

# Custom global context layer
class GlobalContextLayer(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Custom attention mechanism layer
class AttentionMechanism(layers.Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.aspect_matrix = self.add_weight(
            shape=(embedding_dim, embedding_dim),
            initializer=RandomNormal(mean=0.0, stddev=0.05),
            trainable=True,
            name="aspect_matrix"
        )
    def call(self, inputs):
        word_embeddings, global_context = inputs
        intermediate_result = tf.matmul(word_embeddings, self.aspect_matrix)  # (batch_size, max_seq_len, embedding_dim)
        global_context_expanded = tf.expand_dims(global_context, axis=-1)  # (batch_size, embedding_dim, 1)
        attention_scores = tf.matmul(intermediate_result, global_context_expanded)  # (batch_size, max_seq_len, 1)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # (batch_size, max_seq_len, 1)
        weighted_embeddings = attention_weights * word_embeddings  # Element-wise multiplication
        return tf.reduce_sum(weighted_embeddings, axis=1)  # (batch_size, embedding_dim)

    def get_config(self):
        base_config = super().get_config()
        config = {"embedding_dim": self.embedding_dim}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


custom_objects = {
    "GlobalContextLayer": GlobalContextLayer,
    "AttentionMechanism": AttentionMechanism
}