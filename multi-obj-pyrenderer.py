import os
import sys
import random
import argparse
import numpy as np
import trimesh
import pyrender
import util
import imageio


np.random.seed(12354)
random.seed(12354)


p = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
p.add_argument(
    "--data_dir",
    type=str,
    default="/home/nicolai/sra/data/ShapeNetCorev2/03001627/",
    help="Data directory containing shapenet meshes.",
)
p.add_argument(
    "--split",
    type=str,
    required=True,
    help="split from which to pool data",
)
p.add_argument(
    "--output_dir",
    type=str,
    default="./images",
    help="The path the output will be dumped to.",
)
p.add_argument(
    "--num_models",
    type=int,
    default=2,
    help="Number of objects in the scene",
)
p.add_argument(
    "--num_views",
    type=int,
    default=10,
    help="Number of images to render",
)
p.add_argument("--resolution", type=int, default=256, help="output image resolution.")
p.add_argument(
    "--sphere_radius",
    type=float,
    default=1.5,
    help="Radius of the viewing sphere",
)
p.add_argument(
    "--save_png",
    action="store_true",
    help="Save output images for visualization",
)
p.add_argument("--mode", type=str, default="train", help="Options: train and test")


def instance_segmentations(full_depth, renderer, scene, img_size):
    segimg = np.zeros(img_size, dtype=np.uint8)
    flags = pyrender.RenderFlags.DEPTH_ONLY

    # Hide all mesh nodes
    for mn in scene.mesh_nodes:
        mn.mesh.is_visible = False

    # Now iterate through them, enabling each one
    for i, node in enumerate(scene.mesh_nodes):
        node.mesh.is_visible = True
        depth = renderer.render(scene, flags=flags)
        mask = np.logical_and(
            (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0
        )
        segimg[mask] = i + 1
        node.mesh.is_visible = False

    # Show all meshes again
    for mn in scene.mesh_nodes:
        mn.mesh.is_visible = True
    return segimg


def apply_transform(mesh, T, S, R):
    # random rotation of the mesh
    T = trimesh.transformations.translation_matrix(T)
    S = trimesh.transformations.scale_matrix(S)
    M = trimesh.transformations.concatenate_matrices(T, R, S)
    mesh.apply_transform(M)
    return mesh


def get_instance_filenames(data_source, split):
    files = []
    for dataset in split:
        for instance_name in split[dataset]:
            instance_filename = os.path.join(
                dataset, instance_name, "models", "textured.obj"
            )
            if not os.path.isfile(os.path.join(data_source, instance_filename)):
                print("Requested non-existent file '{}'".format(instance_filename))
            files += [instance_filename]
    return files


if __name__ == "__main__":
    args = p.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # with open(splitfile, "r") as f:
    # split = json.load(f)
    instance_names = ["025_mug", "035_power_drill"]

    if len(instance_names) == 0:
        print("Data dir does not contain any instances")
        raise NotImplementedError

    # Load n meshes
    # instance_ids = np.random.choice(
    # np.arange(0, len(instance_names)), args.num_models, replace=True
    # )
    instance_ids = [0, 1]

    trimesh_scene = trimesh.Scene()
    trimesh_meshes = trimesh.Scene()
    translations = [[0.25, 0, 0.15], [-0.3, 0, -0.15]]
    rot_mat = []
    scales = [0.8, 0.95]
    for ii, idx in enumerate(instance_ids):
        mesh = trimesh.load(
            os.path.join(args.data_dir, instance_names[idx], "models", "textured.obj")
        )
        R = trimesh.transformations.rotation_matrix(
            2 * np.random.random(1) * np.pi, [0, 1, 0]
        )
        rot_mat.append(R)
        mesh = apply_transform(mesh, translations[ii], scales[ii], R)
        trimesh_scene.add_geometry(mesh)

        # Do the same again, but force geometries together for instance rendering
        mesh_complete = trimesh.load(
            os.path.join(args.data_dir, instance_names[idx], "models", "textured.obj"),
            force="mesh",
        )

        mesh_complete = apply_transform(mesh_complete, translations[ii], scales[ii], R)
        trimesh_meshes.add_geometry(mesh_complete)

    cam_locations = util.sample_half_sphere(args.num_views, args.sphere_radius)
    obj_location = np.zeros((1, 3))
    cv_poses = util.look_at(cam_locations, obj_location)
    cam_locations = [util.cv_cam2world_to_bcam2world(m) for m in cv_poses]
    image_size = (args.resolution, args.resolution)
    K = np.array([[262.5, 0.0, 128.0], [0.0, 262.5, 128.0], [0.0, 0.0, 1.0]])
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=100
    )

    rgbs = []
    depths = []
    masks = []
    c2ws = []

    scene = pyrender.Scene.from_trimesh_scene(trimesh_scene, ambient_light=(1, 1, 1))
    instance_scene = pyrender.Scene.from_trimesh_scene(
        trimesh_meshes, ambient_light=(1, 1, 1)
    )
    for ii, w2c in enumerate(cam_locations):
        if ii == 0:
            cam_node = scene.add(camera, pose=np.array(w2c))
            cam_node_instance = instance_scene.add(camera, pose=np.array(w2c))
        else:
            scene.set_pose(cam_node, pose=np.array(w2c))
            instance_scene.set_pose(cam_node_instance, pose=np.array(w2c))
        r = pyrender.OffscreenRenderer(*image_size)
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        _, depth_full = r.render(
            instance_scene, flags=pyrender.constants.RenderFlags.FLAT
        )
        mask = instance_segmentations(depth_full, r, instance_scene, image_size)
        w2c = np.array(util.get_world2cam_from_blender_cam(w2c))

        rgbs.append(color)
        depths.append(depth)
        masks.append(mask)
        c2ws.append(np.linalg.inv(w2c))
        r.delete()

        if args.save_png:
            imageio.imwrite(
                os.path.join(args.output_dir, f"example_{str(ii).zfill(3)}.png"),
                color,
            )

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2, 10)
    # for ii in range(10):
    # ax[0, ii].imshow(rgbs[ii])
    # ax[1, ii].imshow(masks[ii], cmap="plasma")
    # plt.show()

    rgb = np.stack([r for r in rgbs])
    depth = np.stack([r for r in depths])
    masks = np.stack([r for r in masks])

    poses = np.stack([r for r in c2ws])

    np.savez_compressed(
        os.path.join(args.output_dir, "example.npz"),
        rgb=rgb,
        depth=depth,
        mask=masks,
        pose=poses,
        K=K,
        R=rot_mat,
        T=translations,
        S=scales,
    )
