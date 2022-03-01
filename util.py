import random
from mathutils import Matrix, Vector
import os
import numpy as np
import math
from functools import reduce

np.seterr(divide="ignore", invalid="ignore")


def depth_2_normal(depth, depth_unvalid, K):
    H, W = depth.shape
    grad_out = np.zeros((H, W, 3))
    X, Y = np.meshgrid(np.arange(0, W), np.arange(0, H))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = ((X - cx) / fx) * depth
    Y = ((Y - cy) / fy) * depth

    XYZ_camera = np.stack([X, Y, depth], axis=-1)

    # compute tangent vectors
    vx = XYZ_camera[1:-1, 2:, :] - XYZ_camera[1:-1, 1:-1, :]
    vy = XYZ_camera[2:, 1:-1, :] - XYZ_camera[1:-1, 1:-1, :]

    # finally compute cross product
    normal = np.cross(vx.reshape(-1, 3), vy.reshape(-1, 3))
    normal_norm = np.linalg.norm(normal, axis=-1)
    normal = np.divide(normal, normal_norm[:, None])

    # reshape to image
    normal_out = normal.reshape(H - 2, W - 2, 3)
    grad_out[1:-1, 1:-1, :] = 0.5 - 0.5 * normal_out

    # zero out +Inf
    grad_out[depth_unvalid] = 0.0
    return grad_out


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


# All the following functions follow the opencv convention for camera coordinates.
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def sample_spherical(n, radius=1.0):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    return xyz


def sample_half_sphere(n, radius, elev=30):
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(math.radians(40), math.radians(80), n)

    x = radius * np.cos(theta) * np.sin(phi)
    z = radius * np.sin(theta) * np.sin(phi)
    y = radius * np.cos(phi)
    return np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)


# Blender: camera looks in negative z-direction, y points up, x points right.
# Opencv: camera looks in positive z-direction, y points down, x points right.
def cv_cam2world_to_bcam2world(cv_cam2world):
    """

    :cv_cam2world: numpy array.
    :return:
    """
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    cam_location = Vector(cv_cam2world[:3, -1].tolist())
    cv_cam2world_rot = Matrix(cv_cam2world[:3, :3].tolist())

    cv_world2cam_rot = cv_cam2world_rot.transposed()
    cv_translation = -1.0 * cv_world2cam_rot @ cam_location

    blender_world2cam_rot = R_bcam2cv @ cv_world2cam_rot
    blender_translation = R_bcam2cv @ cv_translation

    blender_cam2world_rot = blender_world2cam_rot.transposed()
    blender_cam_location = -1.0 * blender_cam2world_rot @ blender_translation

    blender_matrix_world = Matrix(
        (
            blender_cam2world_rot[0][:] + (blender_cam_location[0],),
            blender_cam2world_rot[1][:] + (blender_cam_location[1],),
            blender_cam2world_rot[2][:] + (blender_cam_location[2],),
            (0, 0, 0, 1),
        )
    )

    return blender_matrix_world


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_world2cam_from_blender_cam(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.decompose()[
        0:2
    ]  # Matrix_world returns the cam2world matrix.
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],),
            (0, 0, 0, 1),
        )
    )
    return RT


def cond_mkdir(path):
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def dump(obj):
    for attr in dir(obj):
        if hasattr(obj, attr):
            print("obj.%s = %s" % (attr, getattr(obj, attr)))


def get_archimedean_spiral(sphere_radius, num_steps=250):
    """
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    """
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        z = r * math.sin(-theta + math.pi) * math.sin(-i)
        y = r * -math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)
