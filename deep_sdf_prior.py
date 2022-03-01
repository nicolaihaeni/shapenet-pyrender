# Top of main python script
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import random
import argparse
import numpy as np
import trimesh
import imageio
import open3d as o3d
from mathutils import Matrix

from mesh_to_sdf import get_surface_point_cloud

import pyrender
import util


np.random.seed(0)
random.seed(0)

p = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
p.add_argument(
    "--data_dir",
    type=str,
    default="/home/nicolai/sra/data/ycb/",
    help="Data directory containing meshes.",
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
    default=1,
    help="Number of objects in the scene",
)
p.add_argument(
    "--num_views",
    type=int,
    default=25,
    help="Number of images to render",
)
p.add_argument("--resolution", type=int, default=256, help="output image resolution.")
p.add_argument(
    "--sphere_radius",
    type=float,
    default=1.5,
    help="Radius of the viewing sphere",
)
p.add_argument("--mode", type=str, default="train", help="Options: train and test")
p.add_argument(
    "--dataset", type=str, default="shapenet", help="Options: shapenet and google"
)
p.add_argument(
    "--category", type=str, default="car", help="Options: car, chair, household"
)
p.add_argument(
    "--save_png",
    action="store_true",
    help="Save output images for visualization",
)
p.add_argument(
    "--show_3d",
    action="store_true",
    help="Save output images for visualization",
)
p.add_argument(
    "--proc_id",
    type=int,
    default=0,
    help="process id",
)


def normalize_mesh(mesh):
    # Center the mesh
    matrix = np.eye(4)
    bounds = mesh.bounds
    centroid = (bounds[1, :] + bounds[0, :]) / 2
    matrix[:3, -1] = -centroid
    mesh.apply_transform(matrix)

    # Scale the model to unit diagonal lenght
    matrix = np.eye(4)
    extents = mesh.extents
    diag = np.sqrt(extents[0] ** 2 + extents[1] ** 2 + extents[2] ** 2)
    matrix[:3, :3] *= 1.0 / diag
    mesh.apply_transform(matrix)
    return mesh


def main():
    args = p.parse_args()
    instance_names = []
    train = True

    if args.dataset == "shapenet":
        if args.category == "cars":
            shapenet_categories = ["02958343"]
        elif args.category == "chairs":
            shapenet_categories = ["03001627"]
        elif args.category == "bowl":
            shapenet_categories = ["02880940"]
        elif args.category == "multi-shape":
            shapenet_categories = ["03001627"]
        else:
            if train:
                shapenet_categories = [
                    "04379243",
                    "02958343",
                    "03001627",
                    "02691156",
                    "04256520",
                    "04090263",
                    "03636649",
                    "04530566",
                    "02828884",
                    "03691459",
                    "02933112",
                    "03211117",
                    "04401088",
                ]
            else:
                shapenet_categories = [
                    "02924116",
                    "02808440",
                    "03467517",
                    "03325088",
                    "03046257",
                    "03991062",
                    "03593526",
                    "02876657",
                    "02871439",
                    "03642806",
                    "03624134",
                    "04468005",
                    "02747177",
                    "03790512",
                    "03948459",
                    "03337140",
                    "02818832",
                    "03928116",
                    "04330267",
                    "03797390",
                    "02880940",
                    "04554684",
                    "04004475",
                    "03513137",
                    "03761084",
                    "04225987",
                    "04460130",
                    "02942699",
                    "02801938",
                    "02946921",
                    "03938244",
                    "03710193",
                    "03207941",
                    "04099429",
                    "02773838",
                    "02843684",
                    "03261776",
                    "03759954",
                    "04074963",
                    "03085013",
                    "02834778",
                    "02954340",
                ]

        for cat in shapenet_categories:
            path = os.path.join(args.data_dir, cat)
            instance_names = instance_names + [
                os.path.join(cat, f)
                for f in sorted(os.listdir(path))
                if os.path.isdir(os.path.join(path, f))
            ]
    elif args.dataset == "google" or args.dataset == "ycb":
        instance_names = [
            f
            for f in sorted(os.listdir(args.data_dir))
            if os.path.isdir(os.path.join(args.data_dir, f))
        ]
    else:
        print("Dataset generation for requested dataset not implemented")
        raise NotImplementedError

    instance_names = instance_names[args.proc_id * 8000 :]

    if len(instance_names) == 0:
        print("Data dir does not contain any instances")
        raise NotImplementedError

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Number of files: {len(instance_names)}")

    # Load n meshes
    count = 0
    for instance_name in instance_names:
        runtime_error = False
        if args.dataset == "shapenet":
            category, instance_name = instance_name.split("/")

            if os.path.exists(os.path.join(args.output_dir, f"{instance_name}.npz")):
                continue

            try:
                mesh = trimesh.load(
                    os.path.join(
                        args.data_dir,
                        category,
                        instance_name,
                        "models",
                        "model_normalized.obj",
                    ),
                    force="mesh",
                )
            except ValueError:
                continue
        elif args.dataset == "google":
            if os.path.exists(os.path.join(args.output_dir, f"{instance_name}.npz")):
                continue

            category = "google"
            mesh = trimesh.load(
                os.path.join(args.data_dir, instance_name, "meshes", "model.obj"),
            )
        elif args.dataset == "ycb":
            if os.path.exists(os.path.join(args.output_dir, f"{instance_name}.npz")):
                continue

            category = "ycb"
            mesh = trimesh.load(
                os.path.join(args.data_dir, instance_name, "models", "textured.obj"),
            )

        # Normalize the mesh to unit diagonal
        mesh = normalize_mesh(mesh)

        cam_locations = util.sample_spherical(args.num_views, args.sphere_radius)
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
        normals = []

        scene = pyrender.Scene.from_trimesh_scene(
            trimesh.Scene(mesh), ambient_light=(1, 1, 1)
        )
        for ii, w2c in enumerate(cam_locations):
            # Add camera roll
            theta = random.random() * np.pi
            roll_matrix = Matrix(
                (
                    (np.cos(theta), -np.sin(theta), 0, 0),
                    (np.sin(theta), np.cos(theta), 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                )
            )
            w2c = roll_matrix @ w2c
            if ii == 0:
                cam_node = scene.add(camera, pose=np.array(w2c))
            else:
                scene.set_pose(cam_node, pose=np.array(w2c))

            try:
                r = pyrender.OffscreenRenderer(*image_size)
                color, depth = r.render(
                    scene, flags=pyrender.constants.RenderFlags.FLAT
                )
            except RuntimeError:
                print(f"RuntimeError with instance: {instance_name}. Skipping...")
                runtime_error = True
                break

            normals.append(util.depth_2_normal(depth, depth == 0.0, K))

            mask = depth != 0
            w2c = np.array(util.get_world2cam_from_blender_cam(w2c))

            rgbs.append(color)
            depths.append(depth)
            masks.append(mask)
            c2ws.append(np.linalg.inv(w2c))
            r.delete()

            if args.save_png:
                imageio.imwrite(
                    os.path.join(
                        args.output_dir, f"{instance_name}_{str(ii).zfill(3)}.png"
                    ),
                    color,
                )

        if runtime_error:
            continue

        rgb = np.stack([r for r in rgbs])

        # Check if all images are white. If yes, continue without saving the model
        if np.all(rgb == 255):
            continue

        depth = np.stack([r for r in depths])
        masks = np.stack([r for r in masks])
        poses = np.stack([r for r in c2ws])

        # Generate 3D supervision data for the prior
        number_of_points = 250000
        surface_pcd = get_surface_point_cloud(
            mesh, "scan", args.sphere_radius, 100, 400, 10000000, calculate_normals=True
        )
        pts, sdf = surface_pcd.sample_sdf_near_surface(
            number_of_points,
            1,
            sign_method="normal",
            normal_sample_count=11,
            min_size=0,
            return_gradients=False,
        )
        sdf_pts = np.concatenate([pts, sdf[:, None]], axis=-1)

        if args.show_3d:
            colors = np.zeros_like(pts)
            colors[:, 0] = 1

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            frames = []
            for c in c2ws:
                frames.append(
                    o3d.geometry.TriangleMesh.create_coordinate_frame().transform(c)
                )
            o3d.visualization.draw_geometries(frames + [pcd])

        np.savez_compressed(
            os.path.join(args.output_dir, f"{instance_name}.npz"),
            rgb=rgb,
            depth=depth,
            mask=masks,
            pose=poses,
            K=K,
            sphere_radius=args.sphere_radius,
            sdf=sdf_pts,
            category=category,
            normals=normals,
        )
        count += 1

        if count == 100:
            print(f"Generated {count} new instances")

        if count == 400:
            print("Reaching maximum iterations. Stopping...")
            sys.exit()
    print("Finished all data generation")


if __name__ == "__main__":
    main()
