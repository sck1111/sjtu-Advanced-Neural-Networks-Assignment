# sjtu-Advanced-Neural-Networks-Assignment
Deep Generative Models Evaluation on CIFAR-10 and COVID-19 Radiography Dataset. Systematic evaluation of the generation effectiveness, efficiency, and reliability of multiple deep generation models

The borrowed code base includes:

https://github.com/eriklindernoren/PyTorch-GAN

https://github.com/tkarras/progressive_growing_of_gans

https://github.com/AntixK/PyTorch-VAE

https://github.com/NVlabs/stylegan2

Description

This project evaluates several deep generative models on the CIFAR-10 dataset and a new dataset from Kaggle called "COVID-19 Image Data Collection". The models include DCGAN, VAE, and WGAN-GP. The evaluation focuses on the effectiveness, efficiency, and reliability of the models using metrics like FID, PSNR, SSIM, and training time.

Step 1: Dataset Acquisition

We first use the Kaggle website to find a suitable dataset. For this example, let's choose a recent dataset from the healthcare domain, such as "Breast Cancer Wisconsin (Diagnostic) Data Set".

Other Datasets:

CIFAR-10: CIFAR-10 Dataset

COVID-19 Radiography: COVID-19 Image Data Collection

You can download the dataset from the following URL:

[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

The dataset used is the "Breast Cancer Wisconsin (Diagnostic) Data Set", which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes 569 instances of malignant and benign tumor samples.

COVID-19 Radiography: https://github.com/ieee8023/covid-chestxray-dataset

Step 2: Application of Deep Generative Models

We'll use some popular deep generative models. The goal is to generate synthetic data that mimics the real dataset.

Models

Deep Convolutional Generative Adversarial Networks(DCGAN): A type of neural network designed to generate new data similar to a given dataset.

Variational Autoencoder (VAE): An autoencoder that generates new data by learning the distribution of the input data.

WGAN-GP: Wasserstein GAN with Gradient Penalty

StyleGAN2: Advanced GAN model by NVIDIA


Step 3: Evaluation

We will evaluate the models based on:

Effectiveness: How well the generated data resembles the real data.(FID, PSNR, SSIM, t-SNE visualization.)

Efficiency: Time and computational resources required.(Training time, memory usage.)

Reliability: Consistency of the model's performance.(Consistency across multiple runs, mode collapse detection.)


Evaluation Metrics

Frechet Inception Distance (FID)

Peak Signal-to-Noise Ratio (PSNR)

Structural Similarity Index Measure (SSIM)

Training Time

Step 4: Running Code and Full Report

This completes the runnable code and readme for evaluating deep generative models on the CIFAR-10 and COVID-19 Radiography datasets. The models include DCGAN, VAE, and WGAN-GP. The evaluation metrics are FID, PSNR, SSIM, and training time. The results are summarized and visualized in the end.


Discussion and Answers to Questions

1. Reveal the Problems of Existing Deep Generative Models and Dig into the Causes

Mode Collapse in GANs: GANs, including DCGAN and WGAN-GP, can suffer from mode collapse where the generator produces limited variety of outputs. This occurs because the generator finds a few modes that can consistently fool the discriminator and sticks to generating those.

Blurred Outputs in VAEs: VAEs tend to produce blurred images. This is because the VAE’s loss function, which includes a reconstruction term, encourages the generation of images that are similar to the input in a pixel-wise sense, often leading to less sharp images.

Training Instability: GANs can be unstable to train. This instability is due to the adversarial nature of training two networks against each other, often resulting in oscillations and difficulty in convergence.

2. Algorithmic Improvements to Learning or Reasoning of Existing Deep Generative Models

Improving GANs:

WGAN and WGAN-GP: Implementing the Wasserstein loss with gradient penalty helps improve training stability and reduces mode collapse.

StyleGAN: Introduce improvements in architecture, such as progressively growing the GAN, which has shown to improve training stability and output quality.

Improving VAEs:

Beta-VAE: Adjusting the weight of the KL-divergence term can help control the trade-off between reconstruction fidelity and latent space disentanglement.

VQ-VAE: Vector Quantized VAE introduces a discrete latent space which can help in generating sharper images.

3. New Theoretical Analysis of Existing Deep Generative Models

Theoretical Insights into GANs:

Optimal Transport Theory: WGAN’s foundation is based on optimal transport theory, which provides a robust way to measure the distance between probability distributions.

Nash Equilibrium in GANs: The training of GANs can be framed as a two-player game theoretically, where finding a Nash equilibrium can lead to better understanding of convergence properties.

Information Theory in VAEs:

Information Bottleneck Principle: VAEs can be analyzed using the information bottleneck principle, which helps in understanding how much information from the input is preserved in the latent space.

Bayesian Inference: VAEs provide a probabilistic framework for learning latent representations, connecting with Bayesian inference and allowing for principled uncertainty quantification.

Conclusion

The evaluation of DCGAN, VAE, and WGAN-GP on CIFAR-10 and COVID-19 Radiography datasets revealed strengths and weaknesses of each model. WGAN-GP generally outperformed the others in terms of FID, PSNR, and SSIM, while VAE lagged behind due to its inherent limitations. The analysis and visualization of results help in understanding the performance and guiding further improvements in generative model research.
