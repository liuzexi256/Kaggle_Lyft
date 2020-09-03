

# %% [code] {"_kg_hide-input":true}
import l5kit, os
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
path = '/mnt/g/04_Study/Kaggle_Lyft/'
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = path + "/lyft-motion-prediction-autonomous-vehicles"
# get config
cfg = load_config_data(path + "Lyft_config_files/visualisation_config.yaml")

# %% [markdown]
# Now, let's get a sense of the configuration data. This will include metadata pertaining to the agents, the total time, the frames-per-scene, the scene time and the frame frequency.

# %% [code]
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset
dm = LocalDataManager()
dataset_path = dm.require(cfg["val_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

# %% [markdown]
# Now, however it's time for us to look at the scenes and analyze them in depth. Theoretically, we could create a nifty little data-loader to do some heavy lifting for us.

# %% [code] {"_kg_hide-input":true}
import numpy as np
from IPython.display import display, clear_output
import PIL
 
cfg["raster_params"]["map_type"] = "py_semantic"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 2
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    #display(PIL.Image.fromarray(im[::-1]))

# %% [markdown]
# So, there's a lot of information in this one image. I'll try my best to point everything out, but do notify me if I make any errors. OK, let's get started with dissecting the image:
# + We have an intersection of four roads over here.
# + The green blob represents the AV's motion, and we would require to predict the movement of the AV in these traffic conditions as a sample.

# %% [markdown]
# I don't exactly know what other inferences we can make without more detail on this data, so let's try a satellite-format viewing of these images.

# %% [code]
import numpy as np
from IPython.display import display, clear_output
import PIL
 
cfg["raster_params"]["map_type"] = "py_satellite"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 2
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    #display(PIL.Image.fromarray(im[::-1]))

# %% [markdown]
# Yes! This allows for far more detail than a simple plot without detail. I'd haphazard an educated guess, and make the following inferences:
# + Green still represents the autonomous vehicle (AV), and blue is primarily all the other cars/vehicles/exogenous factors we need to predict for.
# + My hypothesis is that the blue represents the path the vehicle needs to go through.
# + If we are able to accurately predict the path the vehicles go through, it will make it easier for an AV to compute its trajectory on the fly.

# %% [markdown]
# We also want to see how the whole charade of vehicles

# %% [code]
from IPython.display import display, clear_output
from IPython.display import HTML

import PIL
import matplotlib.pyplot as plt
from matplotlib import animation, rc
def animate_solution(images):

    def animate(i):
        im.set_data(images[i])
 
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    
    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)
cfg["raster_params"]["map_type"] = "py_satellite"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 34
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    images.append(PIL.Image.fromarray(im[::-1]))
anim = animate_solution(images)
HTML(anim.to_jshtml())

# %% [markdown]
# So this is a demonstration of the movement of the other vehicles and (in relation to the movement and placement of the other vehicles) the movement of the AV. The AV is currently taking only a straight path in its motion, and a straight path seems logical with the movement and placement of other vehicles.

# %% [code] {"_kg_hide-input":true}
from IPython.display import display, clear_output
from IPython.display import HTML

import PIL
import matplotlib.pyplot as plt
from matplotlib import animation, rc
def animate_solution(images):

    def animate(i):
        im.set_data(images[i])
 
    fig, ax = plt.subplots()
    im = ax.imshow(images[0])
    
    return animation.FuncAnimation(fig, animate, frames=len(images), interval=60)
cfg["raster_params"]["map_type"] = "py_semantic"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 34
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    images.append(PIL.Image.fromarray(im[::-1]))
anim = animate_solution(images)
HTML(anim.to_jshtml())

# %% [markdown]
# We're also able to take a more low-level move by using the semantic option in the Lyft level 5 kit.

# %% [code]
from IPython.display import display, clear_output
import PIL
 
cfg["raster_params"]["map_type"] = "py_semantic"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 34
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    images.append(PIL.Image.fromarray(im[::-1]))
    
anim = animate_solution(images)
HTML(anim.to_jshtml())

# %% [markdown]
# The semantic view is good for a less clustered view but if we want a more detailed, more high-level overview of the data we should perhaps try to use the satellite view voer semantic.

# %% [markdown]
# Now, how about from the agent perspective? This would be quite interesting to consider, as we're modeling from principally the agent perspective in most public notebooks so far.

# %% [code]
import numpy as np
from IPython.display import display, clear_output
import PIL
 
cfg["raster_params"]["map_type"] = "py_satellite"
rast = build_rasterizer(cfg, dm)
dataset = AgentDataset(cfg, zarr_dataset, rast)
scene_idx = 2
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    display(PIL.Image.fromarray(im[::-1]))

# %% [markdown]
# So yes, I probably should save these as a GIF to visualize the agent movements. Let's try a simpler form of this and use the semantic view for the agent dataset.

# %% [code]
import numpy as np
from IPython.display import display, clear_output
import PIL
 
cfg["raster_params"]["map_type"] = "py_semantic"
rast = build_rasterizer(cfg, dm)
dataset = AgentDataset(cfg, zarr_dataset, rast)
scene_idx = 2
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)
    clear_output(wait=True)
    display(PIL.Image.fromarray(im[::-1]))

# %% [markdown]
# We can also take a general view of the street from a matplotlib-type perspective. I borrow this from [this wonderful notebook](https://www.kaggle.com/t3nyks/lyft-working-with-map-api)

# %% [code] {"_kg_hide-input":true}
from l5kit.data.map_api import MapAPI
from l5kit.rasterization.rasterizer_builder import _load_metadata

semantic_map_filepath = dm.require(cfg["raster_params"]["semantic_map_key"])
dataset_meta = _load_metadata(cfg["raster_params"]["dataset_meta_key"], dm)
world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

map_api = MapAPI(semantic_map_filepath, world_to_ecef)
MAP_LAYERS = ["junction", "node", "segment", "lane"]


def element_of_type(elem, layer_name):
    return elem.element.HasField(layer_name)


def get_elements_from_layer(map_api, layer_name):
    return [elem for elem in map_api.elements if element_of_type(elem, layer_name)]


class MapRenderer:
    
    def __init__(self, map_api):
        self._color_map = dict(drivable_area='#a6cee3',
                               road_segment='#1f78b4',
                               road_block='#b2df8a',
                               lane='#474747')
        self._map_api = map_api
    
    def render_layer(self, layer_name):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1])
        
    def render_lanes(self):
        all_lanes = get_elements_from_layer(self._map_api, "lane")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_axes([0, 0, 1, 1])
        for lane in all_lanes:
            self.render_lane(ax, lane)
        return fig, ax
        
    def render_lane(self, ax, lane):
        coords = self._map_api.get_lane_coords(MapAPI.id_as_str(lane.id))
        self.render_boundary(ax, coords["xyz_left"])
        self.render_boundary(ax, coords["xyz_right"])
        
    def render_boundary(self, ax, boundary):
        xs = boundary[:, 0]
        ys = boundary[:, 1] 
        ax.plot(xs, ys, color=self._color_map["lane"], label="lane")
        
        
renderer = MapRenderer(map_api)
fig, ax = renderer.render_lanes()

# %% [code] {"_kg_hide-input":true}
def visualize_rgb_image(dataset, index, title="", ax=None):
    """Visualizes Rasterizer's RGB image"""
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)

    if ax is None:
        fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.imshow(im[::-1])
# Prepare all rasterizer and EgoDataset for each rasterizer
rasterizer_dict = {}
dataset_dict = {}

rasterizer_type_list = ["py_satellite", "satellite_debug", "py_semantic", "semantic_debug", "box_debug", "stub_debug"]

for i, key in enumerate(rasterizer_type_list):
    # print("key", key)
    cfg["raster_params"]["map_type"] = key
    rasterizer_dict[key] = build_rasterizer(cfg, dm)
    dataset_dict[key] = EgoDataset(cfg, zarr_dataset, rasterizer_dict[key])
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, key in enumerate(["stub_debug", "satellite_debug", "semantic_debug", "box_debug", "py_satellite", "py_semantic"]):
    visualize_rgb_image(dataset_dict[key], index=0, title=f"{key}: {type(rasterizer_dict[key]).__name__}", ax=axes[i])
fig.show()

# %% [markdown]
# # ZARR Exploration

# %% [markdown]
# Now that we can explore the images, we can also get a little down and dirty when it comes to the ZARR files. It's rather simple to use with the Python library for exploring them, especially the fact that it's NumPy interoperable.

# %% [code]
print("scenes", zarr_dataset.scenes)
print("scenes[0]", zarr_dataset.scenes[0])


from typing import Dict
#!pip install pytorch-lightning

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path
import pytorch_lightning as pl
import os

path = 'G:/04_Study/Kaggle_Lyft/'
cfg = load_config_data('Lyft_config_files/agent_motion_config.yaml')
class Mod(torch.nn.Module):
    def __init__(self, cfg: Dict):
        super(Mod, self).__init__()
        self.backbone = resnet18(pretrained=False)
        
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        
        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: Bx50x2)
        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)
    def forward(self):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.head(x)
        x = self.logit(x)
        
        return x        

def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs

class LightningLyft(pl.LightningModule):
    def __init__(self, model):
        super(LightningLyft, self).__init__()
        self.model = model
        
    def forward(self, x, *args, **kwargs):
        return self.model(x)
    
    def prepare_train_data(self):
        train_cfg = cfg["train_data_loader"]
        rasterizer = build_rasterizer(cfg, dm)
        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
        train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
        return train_dataloader
            
    def training_step(self, batch, batch_idx):
        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
        losses_train = []
        model = self.model
        for n in [0, 1, 2 , 3, 4]:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
            model.train()
            torch.set_grad_enabled(True)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss(reduction="none")
            loss, _ = forward(data, model, device, criterion)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())
            print(f"LOSS FOR EPOCH {n}: {loss.item()}")
            
    def configure_optimizers(self):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return optimizer

# %% [code]
# ===== INIT DATASET
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])

# %% [code]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Mod(cfg).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="none")

# %% [code]
# ==== TRAIN LOOP
res = []
tr_it = iter(train_dataloader)
model = LightningLyft(build_model(cfg))
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
losses_train = []
for _ in progress_bar:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
    model.train()
    torch.set_grad_enabled(True)
    loss, _ = forward(data, model, device, criterion)
    res.append(_)
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

# %% [markdown]
# First of all, I'm pretty happy that we've been able to train a model with PyTorch lightning since this is the first time I've been using it. Now however, I want to see what exactly this model has learnt. Luckily, we've been able to save predictions into an array `res` for results. Time to check what we've learnt later, but for now let's try a simple prediction.

# %% [markdown]
# Source: https://github.com/lyft/l5kit/blob/master/examples/visualisation/visualise_data.ipynb
# 
# **WORK IN PROGRESS - MORE TO COME.**