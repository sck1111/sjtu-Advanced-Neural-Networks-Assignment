import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(tf.keras.Model):
    def __init__(self, resolution, num_labels):
        super(Discriminator, self).__init__()
        self.num_labels = num_labels
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(resolution, resolution, 1+self.num_labels))
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(0.3)
        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)
        self.resolution = resolution

    def call(self, inputs):
        images, labels = inputs
        labels = tf.expand_dims(labels, axis=1)
        labels = tf.expand_dims(labels, axis=2)
        labels = tf.tile(labels, [1, self.resolution, self.resolution, 1])
        x = tf.concat([images, labels], axis=3)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return self.dense(x)
