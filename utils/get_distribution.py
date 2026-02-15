import pandas as pd
import json

def check_distribution(dataset_path, json_path):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Get distribution
    distribution = df["FraudFound_P"].value_counts().to_dict()

    # Save to JSON file
    with open("json/distribution.json", "w") as f:
        json.dump(distribution, f, indent=4)

    print("Distribution saved to distribution.json")
    print(distribution)


check_distribution("dataset/fraud_oracle.csv", "json/distribution.json")
