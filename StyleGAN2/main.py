import argparse
from train import Trainer
from inference import Inferencer

def main():
    parser = argparse.ArgumentParser(description="Train or Inference")
    parser.add_argument('command', choices=['train', 'inference'], help="Command to execute")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--res', type=int, default=28, help="Resolution")
    parser.add_argument('--config', type=str, default='mnist', help="Configuration")
    args = parser.parse_args()

    if args.command == 'train':
        trainer = Trainer(args.dataset_path, args.batch_size, args.res, args.config)
        trainer.train()
    elif args.command == 'inference':
        inferencer = Inferencer(args.res, args.config)
        inferencer.generate_example()

if __name__ == "__main__":
    main()
