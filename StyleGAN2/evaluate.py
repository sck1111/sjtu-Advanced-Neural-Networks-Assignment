import tensorflow as tf
from modules.generator import Generator
from modules.metrics import FID, IS, PSNR, SSIM
from datasets.mnist import load_mnist

class Evaluator:
    def __init__(self, dataset_path, batch_size, resolution, config):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.resolution = resolution
        self.config = config
        self.dataset, self.num_labels = load_mnist(self.batch_size)
        self.G = Generator(self.resolution, self.num_labels)
        self.G.load_weights('final_generator_weights.h5')  # 载入预训练的生成器模型权重

    def evaluate(self):
        fid = FID(self.G, self.dataset)
        psnr = PSNR(self.G, self.dataset)
        ssim = SSIM(self.G, self.dataset)
        print(f'FID: {fid}, PSNR: {psnr}, SSIM: {ssim}')

if __name__ == "__main__":
    evaluator = Evaluator('./datasets/mnist', 64, 28, 'mnist')
    evaluator.evaluate()
