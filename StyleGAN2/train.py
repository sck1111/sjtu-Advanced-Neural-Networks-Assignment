import tensorflow as tf
from datasets.mnist import load_mnist
from modules.generator import Generator
from modules.discriminator import Discriminator
from modules.metrics import FID, IS, PSNR, SSIM

class Trainer:
    def __init__(self, dataset_path, batch_size, resolution, config):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.resolution = resolution
        self.config = config
       
        self.train_dataset, self.num_labels = load_mnist(self.batch_size)
        self.G = Generator(self.resolution, self.num_labels)
        # self.G.build((None, resolution, resolution, 1+self.num_labels))  # 调用 build 方法
        self.D = Discriminator(self.resolution, self.num_labels)
        self.G_opt = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.999)
        self.D_opt = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5, beta_2=0.999)

        # Compile models
        

       # 调用 build 方法
       #   self.G.build((None, 100+self.num_labels))

    def train_step(self, images, labels):
        # Implement the training step
        noise = tf.random.normal([self.batch_size, 100])
        generated_labels = tf.one_hot(tf.random.uniform([self.batch_size], 0, self.num_labels, dtype=tf.int32), self.num_labels)

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = self.G((noise, generated_labels), training=True)

            real_output = self.D((images, labels), training=True)
            fake_output = self.D((generated_images, generated_labels), training=True)

            disc_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
            disc_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
            disc_loss = disc_loss_real + disc_loss_fake

            gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)

        self.D_opt.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))
        self.G_opt.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))

        return disc_loss, gen_loss

    def train(self):
        for epoch in range(100):  # Example: 100 epochs
            for images, labels in self.train_dataset:
                self.train_step(images, labels)
            print(f'Epoch {epoch+1} completed')

        # Save generator's weights after training
        self.G.save_weights('final_generator_weights.h5')    

    def evaluate(self):
        fid = FID(self.G, self.train_dataset)
        is_score = IS(self.G)
        psnr = PSNR(self.G, self.train_dataset)
        ssim = SSIM(self.G, self.train_dataset)
        print(f'FID: {fid}, IS: {is_score}, PSNR: {psnr}, SSIM: {ssim}')
