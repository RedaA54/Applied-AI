# MNIST Classification with CNN/RNN

## Part 2 of Deep Learning Assignment  
**Objective**: Implement Convolutional Neural Networks (CNN) and/or Recurrent Neural Networks (RNN) for MNIST digit classification, comparing performance against the MLP from Part 1.

---

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib

### Reflection

We discovered that CNNs perform better than MLPs on image data (99.1% vs. 97.8% accuracy) by extracting spatial hierarchies with filters. Visualized the filters of the first layer and discovered they learn edge detectors (see cnn_filters.png), similar to human vision.

Understood the importance of input reshaping for CNNs vs. MLP's ).