import os
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

RAY_DEPTH = True


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        ((i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], np.ones_like(i)), axis=-1
    )
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return np.concatenate((rays_o, rays_d), axis=-1)


base_path = "./images/30b9882d1d75be3537678474be485ca/"
pcd = []

img_files = sorted(glob.glob(os.path.join(base_path, "rgb_*")))
mask_files = sorted(glob.glob(os.path.join(base_path, "mask_*")))
depth_files = sorted(glob.glob(os.path.join(base_path, "depth_*")))
pose_files = sorted(glob.glob(os.path.join(base_path, "pose_*")))

n_views = len(img_files)

K = np.load(os.path.join(base_path, "intrinsics.npy"))
pcds = []
pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.1))
for ii in range(1, n_views):
    img = np.load(img_files[ii]) / 255.0
    mask = np.load(mask_files[ii])
    depth = np.load(depth_files[ii])

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(img)
    # ax2.imshow(mask)
    # ax3.imshow(depth, cmap="plasma")
    # plt.show()

    c2w = np.load(os.path.join(pose_files[ii]))
    H, W = depth.shape

    # Create the point cloud
    if not RAY_DEPTH:
        u = np.arange(0, W)
        v = np.arange(0, H)
        u, v = np.meshgrid(u, v)
        u, v = u.reshape(-1), v.reshape(-1)
        dist = depth.reshape(-1)

        uv = np.stack((u, v, np.ones_like(u)), axis=-1)
        uv = np.linalg.inv(K) @ np.transpose(uv)
        pts = uv * dist[None, :]

        # Transform camera points to world
        pts = c2w @ np.concatenate((pts, np.ones_like(u)[None, :]), axis=0)
        pts = np.transpose(pts)[:, :3]
    else:
        depth = depth.reshape(-1)
        rays = get_rays_np(H, W, K, c2w)

        u = np.arange(0, W)
        v = np.arange(0, H)
        u, v = np.meshgrid(u, v)
        u, v = u.reshape(-1), v.reshape(-1)

        uv = np.stack((u, v, np.ones_like(u)), axis=-1)
        xy = (np.linalg.inv(K) @ np.transpose(uv)) * depth[None, :]
        x = np.transpose(xy)[:, 0]
        y = np.transpose(xy)[:, 1]

        depth = np.sqrt(x ** 2 + y ** 2 + depth ** 2)

        rays = rays.reshape(-1, 6)
        rays0, raysd = rays[:, :3], rays[:, 3:]
        raysd = raysd / np.linalg.norm(raysd, axis=-1, keepdims=True)
        pts = rays0 + raysd * (depth[:, None])

    indices = np.where(mask.reshape(-1))
    pts = pts[indices]
    colors = img.reshape(-1, 3)[indices]
    import open3d as o3d

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1).transform(c2w)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcds.append(pcd)
    pcds.append(frame)
    # o3d.io.write_point_cloud("/home/nicolai/Downloads/plane.ply", pcd)
o3d.visualization.draw_geometries(pcds)
