import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_best_threshold(scores, labels):
    """
    Find threshold with best accuracy
    """

    thresholds = np.linspace(
        scores.min(),
        scores.max(),
        200
    )

    best_threshold = None
    best_accuracy = 0


    for threshold in thresholds:

        predictions = (
            scores > threshold
        ).astype(int)


        accuracy = (
            predictions == labels
        ).mean()


        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold


    return best_threshold, best_accuracy



def plot_histogram(scores, labels, save_path):

    good_scores = scores[labels == 0]
    defect_scores = scores[labels == 1]


    plt.figure(figsize=(8,5))


    plt.hist(
        good_scores,
        bins=50,
        alpha=0.7,
        label="Good"
    )


    plt.hist(
        defect_scores,
        bins=50,
        alpha=0.7,
        label="Defective"
    )


    plt.xlabel(
        "Anomaly Score"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Anomaly Score Distribution"
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()



def main():

    scores_path = Path(
        "outputs/scores.npy"
    )

    labels_path = Path(
        "outputs/labels.npy"
    )


    scores = np.load(
        scores_path
    )

    labels = np.load(
        labels_path
    )


    output_dir = Path(
        "outputs/baseline"
    )

    output_dir.mkdir(
        exist_ok=True
    )


    # Histogram

    plot_histogram(
        scores,
        labels,
        output_dir / "histogram.png"
    )


    # Threshold

    threshold, accuracy = find_best_threshold(
        scores,
        labels
    )


    with open(
        output_dir / "threshold_analysis.txt",
        "w"
    ) as f:

        f.write(
            f"Best Threshold: {threshold:.6f}\n"
        )

        f.write(
            f"Accuracy: {accuracy:.4f}\n"
        )


    print(
        "Visualization completed"
    )

    print(
        f"Best Threshold: {threshold:.6f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )


if __name__ == "__main__":
    main()