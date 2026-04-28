# W-KANet: Wavelet-Guided Kolmogorov-Arnold Network for Cardiac Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"W-KANet: A Wavelet-Guided Kolmogorov-Arnold Network for Robust Cardiac Segmentation in Echocardiographic Images"**.

W-KANet is a novel architecture that synergistically integrates Haar wavelet decomposition, Kolmogorov-Arnold Networks (KANs), and cross-spatial multi-scale attention with tri-stream boundary-reverse fusion for robust cardiac left ventricle segmentation.

## 🏗️ Architecture Overview

<!-- Add your architecture figure here -->
<p align="center">
  <img src="figures/architecture.png" alt="W-KANet Architecture" width="900"/>
</p>

*Figure 1: The complete W-KANet architecture showing the encoder-decoder structure with MSFE, WKGMSABlock, TSBRF, and HDRD components.*

## ✨ Key Features

- **🌊 Wavelet-Guided KAN**: Haar discrete wavelet transform guides dual activation paths in KAN layers
- **🎯 Cross-Spatial Grouped Attention**: Linear complexity O(Nd²) attention mechanism
- **🔄 Tri-Stream Boundary-Reverse Fusion**: Three complementary streams (DAFM + RAM + MBD)
- **📊 Multi-Modal Support**: Echocardiography (binary) and Cardiac MRI (multi-class)
- **⚡ Real-Time Inference**: 15ms per image on RTX 3070 Ti
- **🎨 Task-Adaptive Loss**: Optimized for both binary and multi-class segmentation

## 📋 Requirements

```bash
Python 3.10+
PyTorch 2.0+
CUDA 11.7+ (for GPU training)
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Hassan48khan/W-KANet.git
cd W-KANet

# Create conda environment (recommended)
conda create -n wkanet python=3.10
conda activate wkanet

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets

W-KANet is evaluated on five cardiac imaging datasets:

| Dataset | Modality | Classes | Subjects | Frames |
|---------|----------|---------|----------|--------|
| **CAMUS** | Echocardiography | LV/Myo/LA | 500 | 2,000 |
| **ACDC** | Cardiac MRI | RV/Myo/LV | 150 | 300 |
| **EchoNet-Dynamic** | Echocardiography | LV | 9,989 | 19,978 |
| **HMC-QU** | Echocardiography | Myo | 109 | 2,349 |
| **MCE** | Contrast Echo | Myo | 100 | 3,000 |

## 🚀 Quick Start

### Training

```python
from W_KANet import MSAWKNet
import torch

# Initialize model
model = MSAWKNet(
    num_classes=1,           # 1 for binary, 4 for multi-class (ACDC)
    input_channels=1,        # Grayscale input
    img_size=128,           # Input resolution
    embed_dims=(320, 512),   # Bottleneck dimensions
    gt_ds=True,             # Enable deep supervision
    drop_path_rate=0.1
).cuda()

# Forward pass
x = torch.randn(2, 1, 128, 128).cuda()
aux, main, boundary_maps = model(x)
```

### Inference

```python
# Load pretrained weights
model.load_state_dict(torch.load('checkpoints/wkanet_camus.pth'))
model.eval()

with torch.no_grad():
    output, boundaries = model(input_image)
    prediction = torch.sigmoid(output) > 0.5
```

## 📈 Performance

### Quantitative Results

| Dataset | Dice (%) | IoU (%) | HD95 (mm) |
|---------|----------|---------|-----------|
| **ACDC** | 92.71 | 86.00 | 1.07 |
| **CAMUS** | 93.56 | 87.42 | 3.02 |
| **EchoNet-Dynamic** | 93.91 | 89.47 | 3.04 |
| **HMC-QU** | 94.91 | 89.47 | 2.04 |
| **MCE** | 86.47 | 74.42 | 4.65 |

### Computational Efficiency

| Metric | Value |
|--------|-------|
| **Parameters** | 28.8M |
| **FLOPs** | 45.2G |
| **Inference Time** | 15ms/image |
| **Training Time** | ~10 hours |

## 📁 Repository Structure
W-KANet/
├── W_KANet.py          # Main architecture implementation
├── loss.py             # Task-adaptive loss functions
├── train.py            # Training script
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
├── README.md           # This file
└── figures/
└── architecture.png  # Architecture diagram

## 🧩 Model Components

### 1. Multi-Scale Feature Encoder (MSFE)
Cascaded dilated convolutions with squeeze-and-excitation attention for multi-scale feature extraction.

### 2. Wavelet-Guided KAN with Grouped MSA (WKGMSABlock)
- **Wavelet Decomposition**: Haar DWT separates LL (low-freq) and HH/LH/HL (high-freq) subbands
- **Dual Activation**: LL guides SiLU base path, HH/LH/HL guides B-spline path
- **Grouped Attention**: Linear O(Nd²) complexity through group partitioning

### 3. Tri-Stream Boundary-Reverse Fusion (TSBRF)
- **DAFM**: Dual ASPP fusion for encoder-decoder balancing
- **RAM**: Reverse attention for background suppression
- **MBD**: Multi-scale boundary detection with spatial-channel attention

### 4. Hierarchical Dense Residual Decoder (HDRD)
RSDBlocks with bilinear upsampling for progressive resolution restoration.

## 🎯 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-3 |
| **Weight Decay** | 1e-4 |
| **Batch Size** | 16 |
| **Epochs** | 150 |
| **Scheduler** | Cosine Annealing |
| **Warmup** | 10 epochs |
| **Drop Path** | 0.1 |
| **Image Size** | 128×128 |

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{wkanet2026,
  title={W-KANet: A Wavelet-Guided Kolmogorov-Arnold Network for Robust Cardiac Segmentation in Echocardiographic Images},
  author={Khan, Hassan and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaborations, please contact:
- **Author**: Hassan Khan
- **GitHub**: [@Hassan48khan](https://github.com/Hassan48khan)

## 🙏 Acknowledgments

- UKAN baseline architecture
- BMANet for boundary attention concepts
- BATNet for feature fusion strategies

## ⭐ Star History

If you find this project useful, please consider giving it a star!
