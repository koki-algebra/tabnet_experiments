import os
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer

from utils import get_dataset

# get dataset, categorical indices, dimensions
dataset, cat_idxs, cat_dims = get_dataset(dataset_path="./data/uci_income/adult.csv", target="salary", train_size=0.8)

X_train, y_train = dataset["train"]
X_valid, y_valid = dataset["valid"]

# Notwork parameters
unsupervised_model = TabNetPretrainer(
    cat_idxs=cat_idxs,
    cat_dims=cat_dims,
    cat_emb_dim=3,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="entmax", # sparsemax
    n_shared=1,
    n_independent=1
)

# Self-Supervised Learning
max_epochs = 1000 if not os.getenv("CI", False) else 2

unsupervised_model.fit(
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

# Make reconstruction from a dataset
reconstructed_X, embedded_X = unsupervised_model.predict(X_valid)
assert(reconstructed_X.shape == embedded_X.shape)

unsupervised_explain_matrix, unsupervised_masks = unsupervised_model.explain(X_valid)

# save model
unsupervised_model.save_model('./models/test_pretrain')
