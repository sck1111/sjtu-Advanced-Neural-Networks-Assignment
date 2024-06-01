# sjtu-Advanced-Neural-Networks-Assignment
Systematic evaluation of the generation effectiveness, efficiency, and reliability of multiple deep generation models
The borrowed code base includes:
https://github.com/eriklindernoren/PyTorch-GAN
https://github.com/AntixK/PyTorch-VAE
https://github.com/NVlabs/stylegan2

Description
This project evaluates several deep generative models on the CIFAR-10 dataset. The models include DCGAN, VAE, WGAN-GP, and StyleGAN2. The evaluation focuses on the effectiveness, efficiency, and reliability of the models using metrics like FID, PSNR, SSIM, and training time.

Deep Generative Models Evaluation on Dataset:
Step 1: Dataset Acquisition
We will use the Kaggle website to find a suitable dataset. For this example, let's choose a recent dataset from the healthcare domain, such as "Breast Cancer Wisconsin (Diagnostic) Data Set".

Downloading the Dataset
You can download the dataset from the following URL:
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
The dataset used is the "Breast Cancer Wisconsin (Diagnostic) Data Set", which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes 569 instances of malignant and benign tumor samples.

Step 2: Application of Deep Generative Models
We'll use some popular deep generative models: Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). The goal is to generate synthetic data that mimics the real dataset.
Models
Generative Adversarial Network (GAN): A type of neural network designed to generate new data similar to a given dataset.
Variational Autoencoder (VAE): An autoencoder that generates new data by learning the distribution of the input data.
WGAN-GP: Wasserstein GAN with Gradient Penalty
StyleGAN2: Advanced GAN model by NVIDIA

Evaluation Metrics
Frechet Inception Distance (FID)
Peak Signal-to-Noise Ratio (PSNR)
Structural Similarity Index Measure (SSIM)
Training Time

Step 3: Evaluation
We will evaluate the models based on:
Effectiveness: How well the generated data resembles the real data.(FID, PSNR, SSIM, t-SNE visualization.)
Efficiency: Time and computational resources required.(Training time, memory usage.)
Reliability: Consistency of the model's performance.(Consistency across multiple runs, mode collapse detection.)

Step 4: Running Code and Full Report
Below is the code to apply GANs and VAEs to the dataset, followed by a brief report.
