# CIFAR-100 ResNet with Mixup + EMA

A custom ResNet-18 trained on CIFAR-100 from scratch in PyTorch, featuring research-grade training techniques: Mixup augmentation, Exponential Moving Average (EMA) of model weights, and label smoothing.

Built as a deep learning project while studying CSE at RVCE.

---

## Model Architecture

Custom ResNet-18 adapted for CIFAR-100 (32×32 images):

```
CIFARResNet
├── Stem: Conv2d(3→64, 3×3) + BN + ReLU   (no max-pool, adapted for 32×32)
├── Layer1: 2× BasicBlock(64→64,  stride=1)
├── Layer2: 2× BasicBlock(64→128, stride=2)
├── Layer3: 2× BasicBlock(128→256, stride=2)
├── Layer4: 2× BasicBlock(256→512, stride=2)
├── AdaptiveAvgPool2d → Flatten
└── FC(512 → 100 classes)
```

**BasicBlock** uses zero-initialization on the last BN layer (`nn.init.constant_(bn2.weight, 0)`) — a technique from the "Bag of Tricks" paper that improves early training stability.

---

## Training Techniques

| Technique | Details |
|---|---|
| **Mixup Augmentation** | α=1.0, randomly blends two training images and their labels |
| **EMA (Exponential Moving Average)** | Decay=0.999, separate EMA model used for all validation/testing |
| **Label Smoothing** | ε=0.1 with CrossEntropyLoss |
| **Cosine LR Schedule** | T_max=200 epochs |
| **SGD + Momentum** | lr=0.1, momentum=0.9, weight_decay=5e-4 |
| **Top-1 / Top-5 Accuracy** | Both tracked during validation |

---

## Project Structure

```
vision-project/
├── models/
│   └── resnet.py      # CIFARResNet, BasicBlock, CIFARResNetStem
├── engine/            # Training engine utilities
├── utils/             # Helper functions
├── configs/           # Training configuration
├── train.py           # Main training script
└── README.md
```

---

## Quickstart

```bash
# Install dependencies
pip install torch torchvision scikit-learn numpy

# Train the model (downloads CIFAR-100 automatically)
python train.py
```

The script will:
1. Download CIFAR-100 to `./data/`
2. Train for 200 epochs with Mixup + EMA
3. Save best checkpoint to `best_model.pth`
4. Print top-5 best and worst performing classes

---

## Data Augmentation

```python
train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=(0.5071, 0.4867, 0.4408),
              std=(0.2675, 0.2565, 0.2761))   # CIFAR-100 stats
])
```

---

## Requirements

- Python 3.8+
- PyTorch + torchvision
- scikit-learn (for confusion matrix)
- NumPy

---

## What I Learned

- Building ResNet from scratch (stem, BasicBlock, shortcut connections)
- Zero-init on last BN in residual branch for training stability
- Mixup augmentation and computing the mixed loss
- EMA model averaging — why it outperforms the raw model at test time
- Top-k accuracy computation
- Label smoothing as a regularization technique
