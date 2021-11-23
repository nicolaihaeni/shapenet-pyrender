import os
import sys
import argparse
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import util
import shutil
import cv2
import open3d as o3d


p = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
p.add_argument(
    "--mesh_fpath",
    type=str,
    default="/home/nicolai/sra/data/ShapeNetCorev2/02691156/30b9882d1d75be3537678474be485ca/models/model_normalized.obj",
    help="The path from which to load the mesh.",
)
p.add_argument(
    "--output_dir",
    type=str,
    default="./images",
    help="The path the output will be dumped to.",
)
p.add_argument(
    "--num_observations",
    type=int,
    default=100,
    help="Number of images to render",
)
p.add_argument(
    "--resolution", type=int, default=256, help="The path the output will be dumped to."
)
p.add_argument(
    "--sphere_radius",
    type=float,
    default=1.0,
    help="Radius of the viewing sphere",
)
p.add_argument("--mode", type=str, default="train", help="Options: train and test")


if __name__ == "__main__":
    opt = p.parse_args()

    instance_name = opt.mesh_fpath.split("/")[-3]
    instance_dir = os.path.join(opt.output_dir, instance_name)

    if not os.path.exists(instance_dir):
        util.cond_mkdir(instance_dir)
    else:
        print(f"Folder: {instance_dir} exists already. Skipping")
        sys.exit()

    mesh = trimesh.load(opt.mesh_fpath, force="scene")

    if opt.mode == "train":
        cam_locations = util.sample_spherical(opt.num_observations, opt.sphere_radius)
    elif opt.mode == "test":
        cam_locations = util.get_archimedean_spiral(
            opt.sphere_radius, opt.num_observations
        )

    obj_location = np.zeros((1, 3))
    cv_poses = util.look_at(cam_locations, obj_location)
    cam_locations = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
    image_size = (opt.resolution, opt.resolution)

    K = np.array([[262.5, 0.0, 128.0], [0.0, 262.5, 128.0], [0.0, 0.0, 1.0]])
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=100
    )
    rgbs = []
    depths = []
    masks = []
    c2ws = []

    for ii, w2c in enumerate(cam_locations):
        try:
            scene = pyrender.Scene.from_trimesh_scene(mesh, ambient_light=(1, 1, 1))
        except AttributeError:
            print(f"Instance: {instance_name} failed. Removing folder {instance_dir}")
            os.rmdir(instance_dir)
            sys.exit()

        scene.add(camera, pose=np.array(w2c))
        r = pyrender.OffscreenRenderer(*image_size)
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        mask = (depth != 0).astype(int)
        w2c = np.array(util.get_world2cam_from_blender_cam(w2c))

        rgbs.append(color)
        depths.append(depth)
        masks.append(mask)
        c2ws.append(np.linalg.inv(w2c))
        r.delete()

    rgb = np.stack([r for r in rgbs])
    depth = np.stack([r for r in depths])
    masks = np.stack([r for r in masks])
    poses = np.stack([r for r in c2ws])
    np.savez_compressed(
        os.path.join(instance_dir, f"{instance_name}.npz"),
        rgb=rgb,
        depth=depth,
        mask=masks,
        pose=poses,
        K=K,
    )
