import numpy as np

from utils import get_dataset, rand_nodup

dataset, _, _ = get_dataset(dataset_path="./data/uci_income/adult.csv", target="salary", train_size=0.8)

X_train: np.ndarray
y_train: np.ndarray
X_valid: np.ndarray
y_valid: np.ndarray
X_test: np.ndarray
y_test: np.ndarray

X_train, y_train = dataset["train"]
X_valid, y_valid = dataset["valid"]
X_test, y_test = dataset["test"]

train_size = X_train.shape[0]
valid_size = X_valid.shape[0]
test_size = X_test.shape[0]

total_size = train_size + valid_size + test_size
supervised_size = int(train_size * 0.15)

labeled_indices = rand_nodup(0, train_size-1, supervised_size)
print(X_train[labeled_indices])
print(y_train[labeled_indices])