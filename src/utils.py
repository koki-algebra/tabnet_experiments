from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_dataset(dataset_path: str, target: str, train_size = 0.8) -> Tuple[Dict[str, np.ndarray], List[int], List[int]]:
    if train_size > 1.0 or train_size < 0.0:
        print("train size is invalid")
        train_size = 0.8

    valid_size = (1 - train_size) / 2
    test_size = 1.0 - train_size - valid_size

    target = target

    # read csv
    df = pd.read_csv(dataset_path)
    # split train, valid and test
    if "Set" not in df.columns:
        df["Set"] = np.random.choice(["train", "valid", "test"], p =[train_size, valid_size, test_size], size=(df.shape[0],))    
    train_indices = df[df.Set=="train"].index
    valid_indices = df[df.Set=="valid"].index
    test_indices = df[df.Set=="test"].index


    # Simple preprocessing
    # label encode categorical features and fill empty cells
    nunique = df.nunique()
    types = df.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in df.columns:
        if types[col] == 'object' or nunique[col] < 200:
            l_enc = LabelEncoder()
            df[col] = df[col].fillna("VV_likely")
            df[col] = l_enc.fit_transform(df[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            df[col].fillna(df.loc[train_indices, col].mean(), inplace=True)

    # Define categorical features for categorical embeddings
    unused_feat = ["Set"]
    features = [col for col in df.columns if col not in unused_feat+[target]]

    # indices of categorical features
    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    # dimensions of categorical features
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

    # train
    X_train = df[features].values[train_indices]
    y_train = df[target].values[train_indices]

    # valid
    X_valid = df[features].values[valid_indices]
    y_valid = df[target].values[valid_indices]

    # test
    X_test = df[features].values[test_indices]
    y_test = df[target].values[test_indices]

    dataset = {"train": (X_train, y_train), "valid": (X_valid, y_valid), "test": (X_test, y_test)}

    return dataset, cat_idxs, cat_dims

def rand_nodup(low: int, high: int, size: int) -> np.ndarray:
    if abs(low) + high < size:
        raise ValueError

    r = set()
    while len(r) < size:
        r.add(np.random.randint(low, high))

    return np.array(list(r))
