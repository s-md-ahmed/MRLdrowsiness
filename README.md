# Real-Time Eye State Classification (Awake vs. Sleepy)

A PyTorch-based computer vision pipeline implementing transfer learning with MobileNetV2 and NASNet to classify eye states across ~82,000 images, complete with Exploratory Data Analysis and Grad-CAM interpretability.

---

## Dataset & EDA Insights
- **Total Volume:** ~82,000 images across training, validation, and test splits sourced from the MRL Eye Dataset.
- **Resolution Handling:** Raw exploratory data analysis revealed varying image dimensions (averaging 84x84 pixels), establishing the requirement for strict standardization: mandatory resizing to 224x224, 3-channel RGB conversion, and tensor normalization.

---

## Model Performance Benchmarks

| Model Architecture | Train Accuracy | Val Accuracy | Test Accuracy | Train/Val Gap |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2 (Custom Head)** | 87.27% | 85.75% | 87.00% | 1.52% |
| **NASNet** | 84.00% | 84.00% | 84.00% | ~0.00% |

---

## Model Interpretability & Grad-CAM Visualization

To ensure the model was learning genuine ocular features rather than background lighting or noise, custom Gradient-weighted Class Activation Mapping (Grad-CAM) hooks were integrated targeting the final convolutional layers. This generates spatial heatmaps visually confirming that the model focuses directly on the pupil and eyelid regions.

### Grad-CAM Execution Snippet
```python
from train import show_gradcam, model, val_tf

# Generate and visualize spatial activation heatmaps for real-time validation
show_gradcam(model, "path_to_test_image.png", device="cuda")
