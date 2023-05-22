from keras import layers
from tensorflow import keras


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = keras.Sequential([
            layers.Dense(feed_forward_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        x1 = self.norm1(inputs + attention_output)
        feed_forward_output = self.feed_forward(x1)
        feed_forward_output = self.dropout2(feed_forward_output, training=training)
        return self.norm2(x1 + feed_forward_output)


class TransformerSpeakerIdentification(keras.Model):
    def __init__(self, num_classes, num_layers, embed_dim, num_heads, feed_forward_dim, dropout_rate):
        super(TransformerSpeakerIdentification, self).__init__()
        self.embedding = layers.Embedding(num_classes, embed_dim)
        self.transformer_encoder = [TransformerEncoder(embed_dim, num_heads, feed_forward_dim, dropout_rate)
                                    for _ in range(num_layers)]
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=True):
        x = self.embedding(inputs)
        for encoder in self.transformer_encoder:
            x = encoder(x, training=training)
        x = self.flatten(x)
        return self.dense(x)
