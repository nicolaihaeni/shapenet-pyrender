import os
import glob
import numpy as np
import h5py
import open3d as o3d
import matplotlib.pyplot as plt


files = glob.glob("./shapenet_renderings/*")
for fname in files:
    with h5py.File(fname, "r") as f:
        imgs = f["rgb"][:]
        depths = f["depth"][:]
        normals = f["normals"][:]
        masks = f["mask"][:]
        pose = f["pose"][:]
        K = f["K"][:]
        points = f["sdf"][:]
    f.close()

    normals[np.isnan(normals)] = 0.0
    # Visualize images, normals and depth
    figure, axes = plt.subplots(3, 5)
    for ii in range(5):
        axes[0, ii].imshow(imgs[ii].astype(np.uint8))
        axes[1, ii].imshow(depths[ii], cmap="plasma")
        axes[2, ii].imshow(normals[ii])
    plt.show()

    # Rotate points and project to 0-z plane
    pts = points[:, :3]
    sdf = points[:, 3]
    pts = np.concatenate((pts, np.ones_like(pts[:, -1][:, None])), axis=1)
    pts = np.linalg.inv(pose[1]) @ pts.transpose(1, 0)
    pts = pts.transpose(1, 0)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd = o3d.geometry.PointCloud()
    colors = np.zeros_like(pts[:, :3])
    colors[np.where(sdf <= 0)] = [1, 0, 0]
    colors[np.where(sdf > 0)] = [0, 1, 0]
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([frame, pcd])
