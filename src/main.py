import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

from utils import get_dataset, rand_nodup

# get dataset, categorical indices, dimensions
dataset, cat_idxs, cat_dims = get_dataset(dataset_path="./data/uci_income/adult.csv", target="salary", train_size=0.8)
dataset_name = "UCI Income dataset"

X_train: np.ndarray
y_train: np.ndarray
X_valid: np.ndarray
y_valid: np.ndarray
X_test: np.ndarray
y_test: np.ndarray

X_train, y_train = dataset["train"]
X_valid, y_valid = dataset["valid"]
X_test,  y_test  = dataset["test"]

train_size = X_train.shape[0]
labeled_ratio = 0.05
labeled_size = int(train_size * labeled_ratio)

labeled_indices = rand_nodup(low=0, high=train_size-1, size=labeled_size)

X_l_train = X_train[labeled_indices]
y_l_train = y_train[labeled_indices]


# Self-Supervised Learning
pretrained_model = TabNetPretrainer(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="entmax", # sparsemax
    n_shared=1,
    n_independent=1
)

max_epochs = 1000 if not os.getenv("CI", False) else 2

pretrained_model.fit(
    X_train=X_train,
    eval_set=[X_valid],
    max_epochs=max_epochs,
    patience=5,
    batch_size=2048,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.8
)


# Supervised Learning
clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 10, "gamma": 0.9},
    mask_type="sparsemax"
)

clf.fit(
    X_train=X_l_train,
    y_train=y_l_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=["train", "valid"],
    eval_metric=["auc"],
    max_epochs=max_epochs,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    from_unsupervised=pretrained_model
)


# Test
preds = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_auc}")
