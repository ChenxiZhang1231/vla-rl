
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py

data = "/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/1.0.1/TRI/success/2023-10-31/Tue_Oct_31_13:36:55_2023/trajectory.h5"
trajectory = h5py.File(data, "r")




data = "/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/debug/AUTOLab/2023-09-05/Tue_Sep__5_09:31:13_2023"

img1_path = "/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/debug/AUTOLab/2023-09-05/Tue_Sep__5_09:31:13_2023/images/camera_ext1/left/left_image_step_90.jpg"
img1_dpeth_path = '/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/debug/AUTOLab/2023-09-05/Tue_Sep__5_09:31:13_2023/images/camera_ext1/depth/depth_image_step_90.png'

img2_path = "/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/debug/AUTOLab/2023-09-05/Tue_Sep__5_09:31:13_2023/images/camera_ext2/left/left_image_step_90.jpg"
img2_dpeth_path = '/SSD_DISK/users/zhangjiahui/python-example-droid-dataset/data/debug/AUTOLab/2023-09-05/Tue_Sep__5_09:31:13_2023/images/camera_ext2/depth/depth_image_step_90.png'


gripper_xyz = np.array([0.5646421313285828, 0.2474086731672287, 0.2821006178855896])

intrinsic_matrix = np.array([
    [256.2707305908203, 0.0, 224.01544494628905],
    [0.0, 256.2707305908203, 128.02581939697265],
    [0.0, 0.0, 1.0]
])
extrinsics = {
    "translation": np.array([0.23146988977764021, -0.001996521570894527, 0.4309928622035667]),
    "rotation_matrix": np.array([
        [-0.22670829390835634, 0.874263421332793, 0.42926311231299813],
        [0.9716129331225444, 0.2336072125807258, 0.03736279458592051],
        [-0.06761403450592529, 0.42554804705196675, -0.9024063397317689]
    ])
}

camera_coords = extrinsics["rotation_matrix"].T @ (gripper_xyz - extrinsics["translation"])
image_coords = intrinsic_matrix @ camera_coords
image_coords /= image_coords[2]

img = Image.open(img_path)
plt.imshow(img)
plt.scatter([image_coords[0]], [image_coords[1]], c='red', s=50, marker='o')
plt.title("Gripper Projection on Image")

output_path = "gripper_projection_result.jpg"
plt.savefig(output_path)