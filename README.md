# W-KANet

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation.

## Architecture

<p align="center">
  <img src="Model.png" alt="Architecture" width="900"/>
</p>

## Installation

```bash
git clone https://github.com/Hassan48khan/W-KANet.git
cd W-KANet
pip install -r requirements.txt
```

## Usage

```python
from W_KANet import MSAWKNet
import torch

model = MSAWKNet(num_classes=1, input_channels=1, img_size=128).cuda()
x = torch.randn(2, 1, 128, 128).cuda()
output = model(x)
```

## Files

- `W_KANet.py` - Model architecture
- `loss.py` - Loss functions
- `train.py` - Training script
- `utils.py` - Utilities

## Citation

Details available upon publication.

## License

MIT License

## Contact

Hassan Khan - [@Hassan48khan](https://github.com/Hassan48khan)
