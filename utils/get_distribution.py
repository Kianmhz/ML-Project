import pandas as pd
import json
import os

def check_distribution(dataset_path, json_path, target_col):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Get distribution
    distribution = df[target_col].value_counts().to_dict()

    # Save to JSON file
    with open(json_path, "w") as f:
        json.dump(distribution, f, indent=4)

    print("Distribution saved to distribution.json")
    print(distribution)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "fraud_oracle_preprocessed.csv")
    json_path = os.path.join(base_dir, "json", "distribution.json")
    check_distribution(dataset_path, json_path, "FraudFound_P")
