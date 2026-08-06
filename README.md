# Industrial Anomaly Detection using Convolutional Autoencoder

A PyTorch implementation of an unsupervised anomaly detection system for industrial inspection using the MVTec AD dataset.

The model is trained **only on normal images** and detects anomalies based on image reconstruction error.

---

# Problem

In industrial inspection, collecting defective samples is expensive and often impractical. Instead of learning every possible defect, the model learns the appearance of **normal products**.

During inference:

* Normal images should be reconstructed accurately.
* Defective images should produce larger reconstruction errors.

The reconstruction error is then used as the anomaly score.

---

# Dataset

* Dataset: MVTec AD
* Category: Bottle
* Training Images: Normal samples only
* Test Images:

  * Good
  * Broken Large
  * Broken Small
  * Contamination

Image Size:

```
128 × 128 × 3
```

---

# Model Architecture

```
Input (3×128×128)

↓

Encoder

Conv(3→32)
MaxPool

↓

Conv(32→64)
MaxPool

↓

Conv(64→128)
MaxPool

↓

Conv(128→256)
MaxPool

↓

Latent Space
256×8×8

↓

Decoder

ConvTranspose(256→128)

↓

ConvTranspose(128→64)

↓

ConvTranspose(64→32)

↓

ConvTranspose(32→3)

↓

Sigmoid

↓

Reconstructed Image
```

---

# Training Configuration

| Parameter  |                    Value |
| ---------- | -----------------------: |
| Framework  |                  PyTorch |
| Optimizer  |                     Adam |
| Loss       | Mean Squared Error (MSE) |
| Image Size |                  128×128 |
| Batch Size |                       16 |

---

# Baseline Results

## Reconstruction & ## Error Heatmap


Original image vs reconstructed image.

![Reconstruction](outputs/baseline/comparison.png)
---


---

## Anomaly Score Distribution

Histogram of reconstruction errors for normal and defective samples.

![Anomaly](outputs/baseline/histogram.png)

---

## Sample Scores

| Category        | Label     |   Score |
| --------------- | --------- | ------: |
| Best Good       | Good      | 0.02307 |
| Best Good       | Good      | 0.02328 |
| Best Good       | Good      | 0.02349 |
| Worst Defective | Defective | 0.03873 |
| Worst Defective | Defective | 0.03409 |
| Worst Defective | Defective | 0.03360 |
| Borderline      | Good      | 0.02698 |
| Borderline      | Defective | 0.02690 |

---

# Baseline Analysis

The baseline Autoencoder successfully reconstructs normal images and assigns higher anomaly scores to many defective samples.

However, the score distributions overlap around the decision boundary.

For example:

* Good sample: **0.02698**
* Defective sample: **0.02690**

The difference is extremely small, making threshold-based classification unreliable.

Although the model learns useful visual representations, reconstruction error alone is not sufficient for robust industrial anomaly detection.

---

# Limitations

* Blurry image reconstruction
* Overlapping anomaly score distributions
* Limited separation between normal and defective samples
* Threshold sensitivity

---

# Future Improvements (Version 2)

The next version will improve the latent representation and anomaly detection capability by introducing architectural enhancements such as:

* Batch Normalization
* Skip Connections
* Improved Bottleneck Design
* Better Feature Representation
* Improved Evaluation Metrics

The objective is to increase the separation between normal and defective anomaly scores while preserving reconstruction quality.

---

# Project Structure

```
industrial-autoencoder/

├── data/
├── models/
├── engine/
├── experiments/
├── outputs/
│   ├── checkpoints/
│   └── baseline/
├── utils/
└── README.md
```
