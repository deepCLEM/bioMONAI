import os
import tempfile
from bioMONAI.data import *
from bioMONAI.core import *
from bioMONAI.metrics import accuracy
from bioMONAI.losses import CrossEntropyLossFlat
from monai.networks.nets import DenseNet121
from monai.apps import MedNISTDataset
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
train_ds = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=False)
val_ds = MedNISTDataset(root_dir=root_dir, transform=transform, section="validation", download=False)
test_ds = MedNISTDataset(root_dir=root_dir, transform=transform, section="test", download=False)
data_ops = {
    'x_keys': 'image',
    'y_keys': 'label', 
    'bs': 512,
    'vocab': ['AbdomenCT','BreastMRI','ChestCT','CXR','Hand','HeadCT'],
    'show_summary': True,
}
data = BioDataLoaders.from_monai(train_ds, val_ds, **data_ops)
data.show_batch()
max_epochs = 5
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6)
trainer = fastTrainer(
    data, 
    model, 
    loss_fn=CrossEntropyLossFlat(), 
    metrics=[accuracy], 
    lr=1e-5
)
trainer.fit(max_epochs)
test_dl = test_biodataloader(data, test_ds) 
trainer.show_results(dl=test_dl)