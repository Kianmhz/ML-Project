import pandas as pd
import os
import json

INPUT_PATH = "dataset/fraud_oracle_preprocessed.csv"
OUTPUT_DATASET_DIR = "dataset"
OUTPUT_JSON_DIR = "json"

TARGET_RATIO = 5   # valid : fraud ratio (e.g., 5 = 5:1)
TARGET_COL = "FraudFound_P"
RANDOM_STATE = 42

def downscale_dataset(df, target_col, ratio, random_state=42):
    """
    Downscale majority class to desired ratio.
    ratio = majority : minority
    """

    fraud_df = df[df[target_col] == 1]
    valid_df = df[df[target_col] == 0]

    fraud_count = len(fraud_df)
    desired_valid_count = fraud_count * ratio

    print("Original counts:")
    print(df[target_col].value_counts())

    if desired_valid_count > len(valid_df):
        raise ValueError(
            f"Not enough majority samples to create {ratio}:1 ratio."
        )

    valid_downsampled = valid_df.sample(
        n=desired_valid_count,
        random_state=random_state
    )

    new_df = pd.concat([fraud_df, valid_downsampled])
    new_df = new_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print("\nNew counts:")
    print(new_df[target_col].value_counts())

    return new_df


def create_distribution_json(df, target_col, ratio):
    """
    Create JSON with class counts and percentages
    """

    total = len(df)
    counts = df[target_col].value_counts().to_dict()

    distribution = {
        "dataset_shape": list(df.shape),
        "class_counts": counts,
        "class_percentages": {
            str(k): v / total for k, v in counts.items()
        },
        "ratio_setting": f"{ratio}:1 (valid:fraud)"
    }

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    json_path = os.path.join(
        OUTPUT_JSON_DIR,
        f"distribution_ratio{ratio}.json"
    )

    with open(json_path, "w") as f:
        json.dump(distribution, f, indent=4)

    print(f"Saved distribution: {json_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    new_df = downscale_dataset(
        df,
        TARGET_COL,
        TARGET_RATIO,
        RANDOM_STATE
    )

    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    base_name = os.path.basename(INPUT_PATH).replace(".csv", "")
    output_name = f"{base_name}_ratio{TARGET_RATIO}.csv"
    output_path = os.path.join(OUTPUT_DATASET_DIR, output_name)

    new_df.to_csv(output_path, index=False)
    print(f"\nSaved dataset: {output_path}")

    create_distribution_json(new_df, TARGET_COL, TARGET_RATIO)


if __name__ == "__main__":
    main()
