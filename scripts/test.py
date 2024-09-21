import torch
from unidepth.models import UniDepthV1
import numpy as np
from PIL import Image

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") # or "lpiccinelli/unidepth-v1-cnvnxtl" for the ConvNext backbone



# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open("assets/demo/TimeVideo_20240901_180247_30feet_frame_0000.jpeg"))).permute(2, 0, 1) # C, H, W
# Move the image to the same device as the model (GPU/CPU)
rgb = rgb.to(device)

predictions = model.infer(rgb)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]