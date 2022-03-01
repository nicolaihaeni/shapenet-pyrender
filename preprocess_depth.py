import os
import glob
import numpy as np
import open3d as o3d


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


base_path = "./shapenet_renderings/"
folders = sorted(os.listdir(base_path))

jj = 0
print_n_views = 1000
for folder in folders:
    objects = sorted(os.listdir(os.path.join(base_path, folder)))

    for obj in objects:
        fname = os.path.join(base_path, folder, obj, f"{obj}.npz")
        data = np.load(fname)
        # K = data["K"]
        # rgb = data["rgb"]
        # depth = data["depth"]
        # poses = data["pose"]
        # n_views, H, W = depth.shape

        # # Compute rays in object frame
        # rays = np.stack([get_rays_np(H, W, K, p) for p in poses])
        # # Normalize ray direction
        # rays[..., 3:] = rays[..., 3:] / np.linalg.norm(
        # rays[..., 3:], axis=-1, keepdims=True
        # )

        # for ii in range(n_views):
        # # Transform depth maps to distance maps (distance along ray)
        # dist = depth[ii].reshape(-1)
        # u = np.arange(0, W)
        # v = np.arange(0, H)
        # u, v = np.meshgrid(u, v)
        # u, v = u.reshape(-1), v.reshape(-1)

        # uv = np.stack((u, v, np.ones_like(u)), axis=-1)
        # uv = np.linalg.inv(K) @ np.transpose(uv)
        # xy = uv * dist[None, :]

        # x, y = np.transpose(xy)[..., 0], np.transpose(xy)[..., 1]
        # dist = np.sqrt(x ** 2 + y ** 2 + dist ** 2)
        # depth[ii] = dist.reshape(H, W)

        # indices = np.where(data["mask"][ii].reshape(-1))
        # ray = rays[ii].reshape(-1, 6)
        # rays0, raysd = ray[:, :3], ray[:, 3:]
        # pts = rays0 + raysd * (dist[:, None])
        # pts = pts[indices]
        # img = rgb[ii]
        # colors = img.reshape(-1, 3)[indices]

        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1).transform(
        # poses[ii]
        # )
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # pcds.append(pcd)
        # pcds.append(frame)
        # o3d.visualization.draw_geometries(pcds)

        np.savez_compressed(
            fname,
            rgb=data["rgb"],
            depth=data["depth"],
            mask=data["mask"],
            pose=data["pose"],
            K=data["k"].astype(float),
        )

        if jj % print_n_views == 0:
            print(f"Finished: {jj} examples")
        jj += 1
