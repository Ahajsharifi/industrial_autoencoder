import matplotlib.pyplot as plt
import numpy as np


scores = np.load(
    "outputs/scores.npy"
)

labels = np.load(
    "outputs/labels.npy"
)


plt.hist(
    scores[labels == 0],
    bins=30,
    alpha=0.7,
    label="Normal"
)


plt.hist(
    scores[labels == 1],
    bins=30,
    alpha=0.7,
    label="Anomaly"
)


plt.xlabel("Reconstruction Error")
plt.ylabel("Count")

plt.legend()

plt.show()