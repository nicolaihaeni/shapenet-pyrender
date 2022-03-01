import os
import numpy as np
import trimesh


base_path = "/home/nicolai/sra/data/ycb"
dirs = os.listdir(base_path)
for d in dirs:
    try:
        mesh = trimesh.load(os.path.join(base_path, d, "models", "textured.obj"))
    except AttributeError:
        continue

    extents = mesh.extents
    diag = np.sqrt(extents[0] ** 2 + extents[1] ** 2 + extents[2] ** 2)
    scale = 1.0 / diag
    matrix = np.eye(4)
    matrix[:3, :3] *= scale

    bounds = mesh.bounds
    centroid = bounds[1, :] - ((bounds[1, :] - bounds[0, :]) / 2)
    matrix[:3, -1] = -centroid

    mesh.apply_transform(matrix)

    trimesh.exchange.export.export_mesh(
        mesh, os.path.join(base_path, d, "models", "textured.obj")
    )
