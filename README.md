# 🚗 Transfer Learning on the Road: Self-Driving Car with Vision-Based Control

This project demonstrates a deep learning pipeline that predicts **steering angle** and **speed** of a self-driving car based on grayscale camera input. It uses **DenseNet121** as a pretrained backbone enhanced with **Squeeze-and-Excitation (SE) blocks** for adaptive feature emphasis. The system combines regression and classification tasks and is trained using a custom loss function. The final model is deployed in **TensorFlow Lite (TFLite)** format for real-time, embedded inference.

---

## 📌 Key Highlights

- 🧠 **Transfer Learning with DenseNet121**  
  Fine-tuned on grayscale driving images for dual prediction: speed and steering angle.

- 🔄 **Custom Data Generator**  
  Efficient TensorFlow `Sequence` class for loading and augmenting large datasets on the fly.

- 🎯 **Hybrid Loss Function**  
  Combines **Mean Squared Error** (for steering angle) and **Binary Cross-Entropy** (for speed classification).

- 🧪 **Saliency Map Visualization**  
  Uses gradients to generate saliency overlays, showing which parts of the input image influence decisions.

- 🔧 **Model Optimization**  
  Includes SE blocks and early stopping, learning rate scheduling, and checkpointing for best validation loss.

- 📱 **TFLite Deployment**  
  Model exported to `.tflite` for low-latency inference on embedded hardware.

---

## 📁 Dataset

- 13.8k+ grayscale driving images with corresponding speed and steering angle labels.
- Imbalanced dataset mitigated via **data augmentation (horizontal flipping)**.

---

## 🧠 Model Architecture

```plaintext
Input (90x90 grayscale image)
    ↓
1x1 Conv2D → convert to 3-channel
    ↓
DenseNet121 (pretrained, fine-tuned last 127 layers)
    ↓
Conv2D (512 filters) + SEBlock
    ↓
Flatten → Dense(256)
    ↓
┌──────────────┬──────────────┐
│ Regression   │ Classification │
│ (angle)      │ (speed class)  │
└──────────────┴──────────────┘
```

---

## 🧮 Custom Loss

To handle the dual-task output (steering angle & speed classification), the total loss is computed as:

```python
Total Loss = MSE(steering_angle) + BCE(speed_class)
```

This combined objective ensures that both regression and classification aspects of the model are jointly optimized during training.

---

## 🛰 Deployment Highlights

- ✅ Converted to **TensorFlow Lite (TFLite)** for embedded/real-time inference
- ⚡ Reduced latency from **~700ms** to **~400ms**
- 🧩 Compatibility adjustments made for custom layers (e.g., SEBlock)
- 🧪 Real-world testing on a self-driving car platform
- 🎮 Ensemble voting for prediction stability in varied driving scenarios

---

> 🎓 **Project conducted as part of MSc AI & Data Science at the University of Nottingham**
```
