# Diffusion-Based Generative Modeling for Sparse Calorimeter Detector Data

## Example Generated Shower

![Generated sample](results/generated_sample.png)

## Energy Distribution Comparison

![Energy distribution](results/energy_distribution.png)

---

# Overview

This project explores the use of diffusion-based generative models for simulating sparse calorimeter detector data.

The objective is to evaluate whether **Denoising Diffusion Probabilistic Models (DDPMs)** can reproduce the statistical properties of calorimeter-like particle showers using unlabelled detector events.

The repository implements a complete end-to-end pipeline including:

- dataset preprocessing
- forward diffusion noise injection
- timestep-conditioned denoising network
- reverse diffusion sampling
- physics-aware statistical evaluation

---

# Diffusion Pipeline

The implemented workflow follows the standard DDPM training and generation procedure:
Detector Dataset (HDF5)
↓
Data Preprocessing
↓
Forward Diffusion (noise injection)
↓
Denoising Network (CNN + timestep embedding)
↓
Reverse Diffusion Sampling
↓
Generated Detector Showers
↓
Physics-aware Evaluation

---

# Dataset

The dataset is stored in **HDF5 format** and contains detector events with the following structure:
60000 × 125 × 125 × 8

Where:

- **60000** → number of detector events  
- **125 × 125** → spatial detector grid  
- **8 channels** → detector layers / feature maps  

Each event represents a **sparse energy deposition pattern** inside the detector.

### Dataset Sparsity

- **~98.9% of pixels are zero**
- **~1.1% of pixels contain energy deposits**

This extreme sparsity presents a major challenge for generative models.

---

# Data Processing

The dataset is loaded using **h5py** and converted into PyTorch tensors.

Preprocessing steps:

1. Subset sampling for faster experimentation  
2. Conversion to PyTorch tensors  
3. Channel-first format conversion for CNN compatibility  
4. Normalization to the range **[-1, 1]**

Normalization used during training:
x = x / 255
x = x * 2 - 1

Implementation:
training/dataLoader.py

---

# Model Architecture

A lightweight **convolutional neural network** is used as the diffusion denoiser.

Input shape:
(8, 125, 125)

Network structure:
Conv(8 → 32)
ReLU
Conv(32 → 64)
ReLU
Conv(64 → 32)
ReLU
Conv(32 → 8)

### Timestep Conditioning

The model is conditioned on the diffusion timestep using an embedding layer:
nn.Embedding(T, 32)

The timestep embedding is broadcast across spatial dimensions and injected into the feature maps after the first convolution layer.

The network learns the function:
predicted_noise = f(x_t, t)

Implementation:
model/network.py

---

# Diffusion Process

The implementation follows the **standard DDPM formulation**.

Noise schedule:
beta_t ∈ [1e-4, 0.02]

Forward diffusion:
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

Where:
alpha_t = 1 - beta_t
alpha_bar_t = cumulative product of alpha_t

The neural network is trained to predict the injected noise:
Loss = MSE(predicted_noise, true_noise)

Implementation:
model/diffusion.py

---

# Training

Training configuration:

- diffusion steps: **200**
- batch size: **32**
- optimizer: **Adam**
- learning rate: **1e-4**
- training subset: **5000 samples**
- epochs: **30**

Training procedure:

1. Sample random timestep **t**  
2. Generate noisy input **x_t**  
3. Predict noise **ε**  
4. Compute **MSE loss**  
5. Update model parameters  

Training script:
training/train.py

---

# Sample Generation

New detector events are generated using **reverse diffusion sampling**.

Generation starts from Gaussian noise:
x_T ~ N(0, I)

Reverse process:
for t = T → 1
predict noise
compute x_(t-1)

Generation script:
generate.py

---

# Evaluation

The model is evaluated using **physics-aware statistical comparisons** rather than only visual inspection.

### Total Event Energy Distribution

For each detector event:
total_energy = sum(all detector cells)

This represents the **total energy deposited in the calorimeter**.

Real and generated energy distributions are compared using histograms.

Implementation:
evaluation/energy_distribution.py

Output:
results/energy_distribution.png

---

# Current Limitations

The generated events currently show:

- partial spatial averaging  
- mismatch in total energy distribution compared to real events  

Possible contributing factors include:

- extreme dataset sparsity (~98.9% zero pixels)  
- normalization bias from zero-dominated data  
- limited training subset (5000 samples)  
- relatively lightweight CNN architecture  

These limitations highlight the challenges of applying diffusion models to highly sparse detector data.

---

# Project Structure
diffusion-calorimeter-generation/

inspect/
inspect_data.py
visualize.py

model/
diffusion.py
network.py

training/
dataLoader.py
train.py

evaluation/
energy_distribution.py

results/
generated_sample.png
energy_distribution.png

.gitignore
generate.py
README.md

---

# Future Work

Potential improvements include:

- longer diffusion training
- improved normalization strategies
- spatial shower structure analysis
- additional evaluation metrics:
  - sparsity distribution
  - radial shower profiles
  - Wasserstein distance between distributions

These extensions would help better capture **physical shower geometry** in generated detector events.

---

# Conclusion

This repository demonstrates a **proof-of-concept diffusion-based generative pipeline** for sparse calorimeter detector data.

The project includes:

- dataset preprocessing
- timestep-conditioned diffusion training
- reverse diffusion sample generation
- physics-aware statistical evaluation

The results highlight both the **potential and challenges** of applying diffusion models to sparse detector simulation tasks.
