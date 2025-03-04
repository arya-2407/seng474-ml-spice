"""
Example Usage:

import sys
import os

# Import utils from subfolder of project, works for immediate subfolders of PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) # adjust relative import as necessary
sys.path.append(PROJECT_ROOT)
from utils.data_processing import get_filtered_review_data

X_train, y_train, X_val, y_val, X_test, y_test = get_filtered_review_data('Magazine_Subscriptions', include_columns=['user_id', 'product_id'])
"""

import os
import pickle
import pandas as pd
from datasets import load_dataset

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_COLUMNS = [
    "user_id",
    "product_id",
    "timestamp",
    "title",
    "text",
    "helpful_vote",
]


def _filter_data(df: pd.DataFrame, degeneracy: int):
    """
    - Remove duplicate (user, product) pairs
    - Filter according to graph degeneracy to decrease sparsity.
    """
    df.rename(columns={"parent_asin": "product_id"}, inplace=True)
    df.drop_duplicates(["user_id", "product_id"], inplace=True)

    # Requires multiple passes; users might drop under the threshold after an item they reviewed is removed.
    while True:
        # Check if filtering is complete
        user_counts = df["user_id"].value_counts()
        item_counts = df["product_id"].value_counts()
        if all(user_counts >= degeneracy) and all(item_counts >= degeneracy):
            break

        # Remove sparse connections
        df = df[df["user_id"].isin(user_counts[user_counts >= degeneracy].index)]
        df = df[df["product_id"].isin(item_counts[item_counts >= degeneracy].index)]

    return df.reset_index(drop=True)


def _normalize_data(df: pd.DataFrame):
    """
    - Encode user/product ids as integer indices
    - Normalize ratings linearly on [0,1]
    """
    df["user_id"], _ = pd.factorize(df["user_id"])
    df["product_id"], _ = pd.factorize(df["product_id"])

    # Normalize ratings
    ratings = df["rating"].astype(float)
    min_rating, max_rating = ratings.min(), ratings.max()
    df["rating"] = ratings.apply(
        lambda x: (x - min_rating) / (max_rating - min_rating)
    ).values.astype(float)
    return df


def _split_data(
    df: pd.DataFrame, num_val: int, num_test: int, include_columns: list[str]
):
    """
    - Split data into training, validation, and testing sets
    - Test examples are the last `num_test` ratings.
    - Validation examples are the `num_val` ratings immediately before these.
    """
    # Split the dataset into training, validation, and testing sets
    train_set, val_set, test_set = [], [], []
    for _, user_reviews in df.groupby("user_id"):
        user_reviews = user_reviews.sort_values("timestamp")
        test_set.append(user_reviews.iloc[-num_test:])
        val_set.append(user_reviews.iloc[-num_test - num_val : -num_test])
        train_set.append(user_reviews.iloc[: -num_test - num_val])

    train_set = pd.concat(train_set).reset_index(drop=True)
    val_set = pd.concat(val_set).reset_index(drop=True)
    test_set = pd.concat(test_set).reset_index(drop=True)

    # Separate the features and label
    X_train = train_set.loc[:, include_columns]
    y_train = train_set["rating"]
    X_val = val_set.loc[:, include_columns]
    y_val = val_set["rating"]
    X_test = test_set.loc[:, include_columns]
    y_test = test_set["rating"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_filtered_review_data(
    category: str,
    min_interactions: int = 5,
    include_columns: list[str] = DEFAULT_COLUMNS,
    num_test: int = 1,
    num_val: int = 1,
):
    """
    Processes data from the Amazon Reviews 2023 dataset according to selected category.
    Only keeps features from `include_columns`:
    Options = ["user_id", "product_id", "timestamp", "title", "text", "helpful_vote", "images", "asin", "verified_purchase"]

    Users/items are only included if they have a `min_interactions` count of interactions.
    Train/validation/test are split according to recency of review (test is the newest).

    returns: `X_train, y_train, X_val, y_val, X_test, y_test`
    """
    folder = "data"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{category}_min{min_interactions}_test{num_test}_val{num_val}_cols{include_columns}.pkl"

    # Check if the file already exists
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            print(f"Loading preprocessed data from {filename}")
            return pickle.load(f)

    assert min_interactions > num_test + num_val

    print("Loading raw dataset...")
    dataset = load_dataset(DATASET, "raw_review_" + category, trust_remote_code=True)
    raw_df = dataset["full"].to_pandas()

    print("Processing data...")
    filtered_df = _filter_data(df=raw_df, degeneracy=min_interactions)

    print("Normalizing data...")
    norm_df = _normalize_data(df=filtered_df)

    print("Splitting datasets...")
    # include_columns = list(set(include_columns).union(MANDATORY_COLUMNS))
    splits = _split_data(
        df=norm_df, num_val=num_val, num_test=num_test, include_columns=include_columns
    )
    # X_train, y_train, X_val, y_val, X_test, y_test = splits

    print("Saving...")
    with open(filename, "wb") as f:
        pickle.dump(splits, f)
        print(f"Processed data saved to {filename}")

    return splits
