import tensorflow as tf
from modules.generator import Generator

class Inferencer:
    def __init__(self, resolution, config):
        self.resolution = resolution
        self.config = config
        self.G = Generator(self.resolution)

    def generate_example(self, num_examples=10):
        noise = tf.random.normal([num_examples, 100])
        labels = tf.one_hot(tf.random.uniform([num_examples], 0, 10, dtype=tf.int32), 10)
        images = self.G([noise, labels])
        # Save or display generated images
        for i, image in enumerate(images):
            tf.keras.preprocessing.image.save_img(f'generated_{i}.png', image)
