import numpy as np
import pandas as pd
from pathlib import Path


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


    data = []


    for idx, (score, label) in enumerate(
        zip(scores, labels)
    ):

        data.append(
            {
                "image_id": idx,
                "label": (
                    "Good"
                    if label == 0
                    else "Defective"
                ),
                "score": float(score)
            }
        )


    df = pd.DataFrame(data)


    # -------------------------
    # 1. Best normal samples
    # -------------------------

    best_good = (
        df[df["label"] == "Good"]
        .sort_values(
            "score"
        )
        .head(5)
    )


    # -------------------------
    # 2. Worst defective samples
    # -------------------------

    worst_defective = (
        df[df["label"] == "Defective"]
        .sort_values(
            "score",
            ascending=False
        )
        .head(5)
    )


    # -------------------------
    # 3. Borderline samples
    # -------------------------

    threshold = df["score"].median()


    borderline = (
        df.assign(
            distance=abs(
                df["score"] - threshold
            )
        )
        .sort_values(
            "distance"
        )
        .head(5)
    )


    result = pd.concat(
        [
            best_good.assign(
                category="Best Good"
            ),

            worst_defective.assign(
                category="Worst Defective"
            ),

            borderline.assign(
                category="Borderline"
            )
        ]
    )


    result = result[
        [
            "category",
            "image_id",
            "label",
            "score"
        ]
    ]


    result.to_csv(
        output_dir / "score_examples.csv",
        index=False
    )


    print(
        "Saved:",
        output_dir / "score_examples.csv"
    )

    print(result)



if __name__ == "__main__":
    main()