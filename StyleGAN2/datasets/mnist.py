import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_mnist(batch_size):
    (train_images, train_labels), _ = mnist.load_data()
    train_images = train_images[..., tf.newaxis].astype("float32") / 255.0
    train_labels = tf.one_hot(train_labels, depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    return train_dataset, 10
