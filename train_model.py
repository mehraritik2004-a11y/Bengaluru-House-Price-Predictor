"""
train_model.py
---------------
Standalone training script for the Bengaluru House Price Predictor.

Run this ONCE (locally, with the exact package versions pinned in
requirements.txt) to regenerate `house_price_model.pkl` and
`locations.json`. This avoids the #1 cause of Streamlit Cloud deploy
failures: a model pickled with one scikit-learn/pandas/numpy version
being loaded with a different version on the server.

Usage:
    pip install -r requirements.txt
    python train_model.py
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "Bengaluru_House_Data.csv"


def convert_range(x):
    temp = x.split("-")
    if len(temp) == 2:
        return (float(temp[0]) + float(temp[1])) / 2
    try:
        return float(x)
    except ValueError:
        return None


def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for _, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_output = pd.concat([df_output, gen_df], ignore_index=True)
    return df_output


def bhk_outlier_remover(df):
    exclude_indices = np.array([])
    for _, location_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0],
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < (stats["mean"])].index.values,
                )
    return df.drop(exclude_indices, axis="index")


def main():
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)

    # Drop unneeded columns
    data.drop(columns=["area_type", "availability", "society", "balcony"], inplace=True)

    # Fill nulls
    data["location"] = data["location"].fillna("Whitefield")
    data["size"] = data["size"].fillna("2 BHK")
    data["bath"] = data["bath"].fillna(data["bath"].median())

    # bhk column from size
    data["bhk"] = data["size"].str.split().str.get(0).astype(int)

    # total_sqft ranges -> float
    data["total_sqft"] = data["total_sqft"].apply(convert_range)
    data.dropna(subset=["total_sqft"], inplace=True)

    data.drop(columns=["size"], inplace=True)

    # price per sqft (used only for outlier removal, dropped afterwards)
    data["price_per_sqft"] = data["price"] * 100000 / data["total_sqft"]

    # Clean location strings, bucket rare locations as "other"
    data["location"] = data["location"].apply(lambda x: x.strip())
    location_count = data["location"].value_counts()
    location_count_less_10 = location_count[location_count <= 10]
    data["location"] = data["location"].apply(
        lambda x: "other" if x in location_count_less_10 else x
    )

    # Outlier removal
    data = data[((data["total_sqft"] / data["bhk"]) >= 300)]
    data = remove_outliers_sqft(data)
    data = bhk_outlier_remover(data)

    data.drop(columns=["price_per_sqft"], inplace=True)
    data.to_csv("Cleaned_data.csv", index=False)
    print(f"Cleaned data shape: {data.shape}")

    X = data.drop(columns=["price"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    column_trans = make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["location"]),
        remainder="passthrough",
    )
    scaler = StandardScaler()
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, scaler, lr)

    print("Training model...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(f"R2 score : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f} Lakhs")

    with open("house_price_model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    locations = sorted(data["location"].unique().tolist())
    with open("locations.json", "w") as f:
        json.dump(locations, f)

    print("Saved house_price_model.pkl and locations.json")


if __name__ == "__main__":
    main()
