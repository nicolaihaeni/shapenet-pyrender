import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


file_name = "/home/nicolai/sra/code/shapenet-pyrender/ycb/022_windex_bottle.npz"
data = np.load(file_name)

img = data["rgb"][0]
fig, ax = plt.subplots(1, 1)
ax.imshow(img)

pts = data["sdf"][:, :3]
pose = data["pose"][0]
frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

colors = np.zeros_like(pts)
colors[:, 0] = 1

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.transform(np.linalg.inv(pose))

o3d.visualization.draw_geometries([frame, pcd])
plt.show()
