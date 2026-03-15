import logging
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
import torch
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import LoadImageD, EnsureChannelFirstD, ScaleIntensityD, Compose
print_config()
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
transform = Compose(
    [
        LoadImageD(keys="image", image_only=True),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
    ]
)
dataset = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True)
max_epochs = 5
device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
model = densenet121(spatial_dims=2, in_channels=1, out_channels=6).to(device)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
trainer = SupervisedTrainer(
    device=device,
    max_epochs=max_epochs,
    train_data_loader=DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4),
    network=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5),
    loss_function=torch.nn.CrossEntropyLoss(),
    inferer=SimpleInferer(),
    train_handlers=StatsHandler(),
)
trainer.run()
dataset_dir = Path(root_dir, "MedNIST")
class_names = sorted(f"{x.name}" for x in dataset_dir.iterdir() if x.is_dir())
testdata = MedNISTDataset(root_dir=root_dir, transform=transform, section="test", download=False, runtime_cache=True)
max_items_to_print = 10
with eval_mode(model):
    for item in DataLoader(testdata, batch_size=1, num_workers=0):
        prob = np.array(model(item["image"].to(device)).detach().to("cpu"))[0]
        pred = class_names[prob.argmax()]
        gt = item["class_name"][0]
        print(f"Class prediction is {pred}. Ground-truth: {gt}")
        max_items_to_print -= 1
        if max_items_to_print == 0:
            break