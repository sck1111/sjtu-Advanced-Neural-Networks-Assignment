import argparse
from train import Trainer
from inference import Inferencer

def main():
    parser = argparse.ArgumentParser(description="StyleGAN2 MNIST Training and Inference")
    subparsers = parser.add_subparsers(dest='command')
    
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--dataset_path', required=True, type=str, help='Path to the dataset')
    train_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--res', type=int, required=True, help='Resolution')
    train_parser.add_argument('--config', type=str, required=True, help='Config name')
    
    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument('--res', type=int, required=True, help='Resolution')
    inference_parser.add_argument('--config', type=str, required=True, help='Config name')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        trainer = Trainer(args.dataset_path, args.batch_size, args.res, args.config)
        trainer.train()
    elif args.command == 'inference':
        # inferencer = Inferencer(args.res, args.config)
        # inferencer.generate_example()
        # Assuming you have the inferencer set up correctly
        inferencer = Inferencer(28, 'mnist')

# Generate some noise and labels for inference
        noise = tf.random.normal([1, 100])
        labels = tf.one_hot([5], 10)  # Example label

# Generate an image
        generated_image = inferencer.G((noise, labels))

# Display or save the generated image to verify the result
        print("Generated image shape:", generated_image.shape)

if __name__ == "__main__":
    main()
