import os
import torch
from sklearn.metrics import accuracy_score

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

from utils import get_dataset

# get dataset, categorical indices, dimensions
dataset, cat_idxs, cat_dims = get_dataset(dataset_path="./data/uci_income/adult.csv", target="salary", train_size=0.8, labeled_ratio=0.15)
dataset_name = "UCI Income dataset"

X_l_train, y_l_train = dataset["train_labeled"]
X_u_train, _         = dataset["train_unlabeled"]
X_valid,   y_valid   = dataset["valid"]
X_test,    y_test    = dataset["test"]

# ----- Pre-training -----

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
    X_train=X_u_train,
    eval_set=[X_valid],
    max_epochs=max_epochs,
    patience=5,
    batch_size=2048,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.8
)

# ----- fine-tuning -----

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
    eval_set=[(X_l_train, y_l_train), (X_valid, y_valid)],
    eval_name=["train", "valid"],
    eval_metric=["accuracy"],
    max_epochs=max_epochs,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    from_unsupervised=pretrained_model
)


# ----- test -----

# Accuracy
pred_test = clf.predict(X_test)
test_acc = accuracy_score(y_pred=pred_test, y_true=y_test)

print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")
print(f"FINAL TEST SCORE FOR {dataset_name} : {test_acc}")
