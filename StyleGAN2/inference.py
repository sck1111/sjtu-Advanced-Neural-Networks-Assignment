import tensorflow as tf
from modules.generator import Generator
import h5py






class Inferencer:
    def __init__(self, resolution, config):
        self.resolution = resolution
        self.config = config
        self.G = Generator(self.resolution, 10)
        # 在这里调用一次模型，以创建模型的变量
        #_ = self.G((tf.zeros((1, 100)), tf.zeros((1, 10))))
        # inferencer = Inferencer(args.res, args.config)
        # inferencer.G.load_weights('final_generator_weights.h5')
        # self.G.load_weights('final_generator_weights.h5')
        # Load weights using h5py
        # Ensure model is built
        noise = tf.random.normal([1, 100])
        labels = tf.one_hot([0], 10)
        _ = Generator((noise, labels))  # 调用模型以初始化权重
        #noise = tf.random.normal([1, 100])
        #labels = tf.random.uniform([1, 10], minval=0, maxval=10, dtype=tf.int32)
        self.G((noise, labels))

        # Load weights using h5py
        with h5py.File('final_generator_weights.h5', 'r') as f:
            for layer in self.G.layers:
                if layer.name in f:
                    print(f"Loading weights for layer: {layer.name}")
                    g = f[layer.name]
                    weights = [np.array(g[weight]) for weight in g.attrs['weight_names']]
                    print(f"Weights loaded for layer {layer.name}: {weights}")
                    layer.set_weights(weights)

        print("Weights loaded successfully")
    def generate_example(self, num_examples=10):
        noise = tf.random.normal([num_examples, 100])
        labels = tf.one_hot(tf.random.uniform([num_examples], 0, 10, dtype=tf.int32), 10)
        # noise = tf.zeros((1, 100))
        # labels = tf.zeros((1, 10))
        images = self.G((noise, labels))
        # noise_and_labels = (tf.zeros((1, 100)), tf.zeros((1, 10)))
        # images = self.G(noise_and_labels)

        for i, image in enumerate(images):
            tf.keras.preprocessing.image.save_img(f'generated_{i}.png', image)

# if __name__ == "__main__":
#     inferencer = Inferencer(28, 'mnist')
#     inferencer.generate_example()