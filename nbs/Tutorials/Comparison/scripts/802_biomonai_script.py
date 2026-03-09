import os
import shutil
import tempfile
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bioMONAI.data import *
from bioMONAI.core import *
from bioMONAI.metrics import ROCAUCMetric
from bioMONAI.datasets import download_file
from bioMONAI.core import parent_label, Path, accuracy
from bioMONAI.losses import CrossEntropyLossFlat
from monai.config import print_config
from monai.networks.nets import DenseNet121
from monai.transforms import LoadImageD, EnsureChannelFirstD, ScaleIntensityD, Compose
from monai.apps import MedNISTDataset
print_config()
base_directory = '../../_data/'
if base_directory is not None:
    os.makedirs(base_directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if base_directory is None else base_directory
print(root_dir)
transform = Compose(
    [
        LoadImageD(keys="image", image_only=True),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
    ]
)
train_df = pd.DataFrame(MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True).data)
val_df = pd.DataFrame(MedNISTDataset(root_dir=root_dir, transform=transform, section="validation", download=False, runtime_cache=True).data)
test_df = pd.DataFrame(MedNISTDataset(root_dir=root_dir, transform=transform, section="test", download=False, runtime_cache=True).data)
full_train_df = pd.concat([train_df.assign(is_valid=0), val_df.assign(is_valid=1)], ignore_index=True)
data_ops = {
    'fn_col': ['image'],
    'label_col': ['class_name'],
    'valid_col': ['is_valid'],
    'bs': 512,
    'shuffle': True,
}
data = BioDataLoaders.class_from_df(full_train_df, **data_ops)
test_data = test_biodataloader(data, test_df) # type:ignore
max_epochs = 5
device = get_device()
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6)
loss_function = CrossEntropyLossFlat()
metrics = [accuracy]
trainer = fastTrainer(
    data, 
    model, 
    loss_fn=loss_function, 
    metrics=metrics, 
    lr=1e-5
)
trainer.fit(max_epochs)
evaluate_classification_model(trainer, test_data, metrics=accuracy, show_graph=True);