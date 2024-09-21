import torch
from unidepth.models import UniDepthV1
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Load the UniDepth model
model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

print(torch.cuda.is_available()) 

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Load the RGB image and move to the GPU
rgb = torch.from_numpy(np.array(Image.open("assets/demo/TimeVideo_20240901_180247_30feet_frame_0000.jpeg"))).permute(2, 0, 1)  # C, H, W
rgb = rgb.to(device)  # Move to GPU
intrinsics_torch = torch.from_numpy(np.load("assets/demo/camera_matrix.npy"))

# Convert the intrinsics matrix to float32
intrinsics_torch = intrinsics_torch.float()

# Perform inference
predictions = model.infer(rgb, intrinsics_torch)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]

# Squeeze to remove unnecessary dimensions and move to CPU
depth_map = depth.squeeze().cpu().numpy()

# Visualize the depth map
# plt.imshow(depth_map, cmap='plasma')
# plt.colorbar()
# plt.title('Predicted Depth Map')
# plt.show()

# Load the car pixel positions from the file
def load_pixel_positions(filepath):
    pixel_positions = []
    with open(filepath, 'r') as file:
        for line in file:
        # Skip the header line
            if line.startswith("#"):
                continue
            x, y = map(int, line.strip().split())
            pixel_positions.append((x, y))
    return pixel_positions

# Extract depth values at specific pixel positions
def extract_depth_values(depth, pixel_positions):
    depth_values = []
    for x, y in pixel_positions:
        depth_values.append(depth[0, 0, y, x].item())  # Assuming depth shape is [1, 1, H, W]
    return depth_values

# Path to your pixel positions file
pixel_positions_filepath = "./assets/demo/car_pixel_positions_test_car_w_timestamp.txt"
pixel_positions = load_pixel_positions(pixel_positions_filepath)

# Extract depth values at the specified pixel positions
depth_values = extract_depth_values(depth, pixel_positions)

# Print the depth values for the specific pixel positions
# for idx, depth_value in enumerate(depth_values):
#     print(f"Pixel {pixel_positions[idx]}: Depth = {depth_value}")

# Load the RGB image
rgb_image = Image.open("assets/demo/TimeVideo_20240901_180247_30feet_frame_0000.jpeg")
rgb_np = np.array(rgb_image)

# Plot the original image
# plt.imshow(rgb_np)

# Mark the specific pixel positions on the image
y_coords, x_coords = zip(*pixel_positions)
# plt.scatter(x_coords, y_coords, c='red', s=0.5, label="Car Pixels")

# # Display the image with marked pixels
# plt.title("Car Pixels on the Original Image")
# plt.legend()
# plt.show()

# Remove the batch dimension, currently shape is [1, 3, 1080, 1920]
xyz = xyz.squeeze(0)  # Now shape is [3, 1080, 1920]

# Extract the 3D coordinates for the corresponding pixel positions
point_cloud = []
for (x, y) in pixel_positions:
    if 0 <= x < xyz.shape[2] and 0 <= y < xyz.shape[1]:
        # Extracting the point cloud at (x, y). Now xyz is [3, H, W]
        point_3d = xyz[:, y, x]  # (3,) extracting the 3D point (X, Y, Z)
        point_cloud.append(point_3d.cpu().numpy())
    else:
        print(f"Pixel ({x}, {y}) is out of bounds.")

# Convert the list of points into a numpy array
point_cloud = np.array(point_cloud)

# Visualize the 3D point cloud if the shape is correct
if point_cloud.shape[1] == 3:  # Ensure we have (N, 3) points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Split the points into X, Y, Z coordinates
    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    zs = point_cloud[:, 2]

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=2,
            color=zs,                # Set color to the Z coordinate values
            colorscale='Jet',         # Color scale
            opacity=0.8
        )
    )])
    
    # Update layout for better visual
    fig.update_layout(scene=dict(
                        xaxis_title='X Coordinate',
                        yaxis_title='Y Coordinate',
                        zaxis_title='Z Coordinate'),
                      margin=dict(r=0, b=0, l=0, t=0))
    
    # Show the plot (this opens an interactive window in Jupyter, or a browser if running standalone)
    fig.show()
else:
    print("Point cloud does not have 3D points.") 