# Real-Time Eye State Classification (Awake vs. Sleepy)

A PyTorch-based computer vision pipeline implementing transfer learning with MobileNetV2 and NASNet to classify eye states across ~82,000 images, complete with Exploratory Data Analysis and Grad-CAM interpretability.

---

## Dataset & EDA Insights
- **Total Volume**: ~82,000 images total across training, validation, and test splits.
- **Source**: MRL Eye Dataset.
- **Resolution Handling**: Raw image analysis revealed varying dimensions (avg 84x84), establishing the requirement for standardisation via mandatory `224x224` resizing, 3-channel RGB grayscale conversion, and normalization.

---

## Model Performance Benchmarks

| Model Architecture | Train Accuracy | Val Accuracy | Test Accuracy | Train/Val Gap |
| :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Custom Head)** | **87.27%** | **85.75%** | **87.00%** | **1.52%** |
| **NASNet** | 84.00% | 84.00% | 84.00% | ~0.00% |

---

## Model Interpretability (Grad-CAM)
Integrated Class Activation Mapping hooks targeting the final convolutional layers to generate spatial heatmaps, confirming the model focuses on relevant eye regions rather than background artifacts.

```python
from train import show_gradcam, model, val_tf

show_gradcam(model, "path_to_test_image.png", device="cuda")
