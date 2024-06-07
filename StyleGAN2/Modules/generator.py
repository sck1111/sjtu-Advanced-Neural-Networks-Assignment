import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, resolution, num_labels):
        super(Generator, self).__init__()
        self.num_labels = num_labels
        # self.dense1 = layers.Dense(7*7*256, use_bias=False,)
        self.dense1 = layers.Dense(7*7*256, use_bias=False, input_shape=(100+self.num_labels,))
        self.batch_norm1 = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.reshape = layers.Reshape((7, 7, 256))
        self.conv2d_transpose1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batch_norm2 = layers.BatchNormalization()
        self.conv2d_transpose2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm3 = layers.BatchNormalization()
        self.conv2d_transpose3 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

    def call(self, inputs):
        noise, labels = inputs
        labels = tf.cast(labels, tf.float32)  # Convert labels to float32
        x = tf.concat([noise, labels], axis=1)
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.reshape(x)
        x = self.conv2d_transpose1(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu(x)
        x = self.conv2d_transpose2(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu(x)
        return self.conv2d_transpose3(x)
