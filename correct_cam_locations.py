import os
import glob
import numpy as np
import util
from mathutils import Matrix


all_files = glob.glob("./images/**/*.npz", recursive=True)

for fname in all_files:
    data = np.load(fname)
    poses = data["pose"]
    for ii, w2c in enumerate(poses):
        mat = Matrix(w2c.tolist())
        w2c = util.np.array(util.get_world2cam_from_blender_cam(mat))
        poses[ii] = np.linalg.inv(w2c)
    np.savez_compressed(
        fname,
        rgb=data["rgb"],
        depth=data["depth"],
        mask=data["mask"],
        pose=poses,
        K=data["K"],
    )
