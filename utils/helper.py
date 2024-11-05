from __future__ import annotations

import collections
import json
import math
import typing as t
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class Colmap_Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray
    extrinsic: np.ndarray | None = None


BaseImage = collections.namedtuple(
    "BaseImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def _read_slam_intrinsic_and_extrinsic(json_path: Path | str, image_folder: Path | str,
                                       output_convention: t.Literal["opencv", "slam"] = "opencv", ) -> t.Tuple[
    t.Dict[int, Colmap_Camera], t.Dict[int, Image]]:
    json_path = Path(json_path)
    image_folder = Path(image_folder)
    assert json_path.exists(), f"Path {json_path} does not exist."
    assert (image_folder.exists() and image_folder.is_dir()), f"Path {image_folder} does not exist."

    with open(json_path, "r") as f:
        meta_file = json.load(f)

    camera_calibrations = meta_file["calibrationInfo"]
    available_cam_list = list(camera_calibrations.keys())
    cameras = {}

    for camera_id, (cur_camera_name, camera_detail) in enumerate(camera_calibrations.items(), start=1):
        model = "PINHOLE"
        width = camera_detail["intrinsics"]["width"]
        height = camera_detail["intrinsics"]["height"]
        params = [camera_detail["intrinsics"]["camera_matrix"][0], camera_detail["intrinsics"]["camera_matrix"][4],
                  camera_detail["intrinsics"]["camera_matrix"][2], camera_detail["intrinsics"]["camera_matrix"][5], ]
        params = np.array(params).astype(float)
        extrinsic = np.array(
            [camera_detail["qw"], camera_detail["qx"], camera_detail["qy"], camera_detail["qz"], camera_detail["x"],
             camera_detail["y"], camera_detail["z"], ])

        cameras[camera_id] = Colmap_Camera(id=camera_id, model=model, width=width, height=height, params=params,
                                           extrinsic=extrinsic)

    # del camera_id

    def iterate_word2cam_matrix(meta_file, image_folder):
        available_image_names = [x.relative_to(image_folder).as_posix() for x in
                                 chain(image_folder.rglob("*.png"), image_folder.rglob("*.jpeg"))]

        for cur_frame in meta_file["data"]:
            for cur_camera_name, cur_c2w in cur_frame["worldTcam"].items():
                if cur_camera_name in cur_frame["imgName"]:
                    if cur_frame["imgName"][cur_camera_name] in available_image_names:
                        yield cur_frame["imgName"][cur_camera_name], cur_camera_name, cur_c2w

    def quaternion_to_rotation_matrix(q):
        w, x, y, z = q
        return np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w, ],
                         [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w, ],
                         [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2, ], ])

    S = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)

    images = {}
    extrinsic_raw = list(iterate_word2cam_matrix(meta_file, image_folder))
    logger.info(f"Found {len(extrinsic_raw)} images in the meta file.")

    for image_id, (cur_name, cur_camera_name, cur_frame) in enumerate(extrinsic_raw):
        px = cur_frame["px"]
        py = cur_frame["py"]
        pz = cur_frame["pz"]
        qw = cur_frame["qw"]
        qx = cur_frame["qx"]
        qy = cur_frame["qy"]
        qz = cur_frame["qz"]
        R = np.zeros((4, 4))
        qvec = np.array([qw, qx, qy, qz])
        norm = np.linalg.norm(qvec)
        qvec /= norm
        R[:3, :3] = quaternion_to_rotation_matrix(qvec)
        R[:3, 3] = np.array([px, py, pz])
        R[3, 3] = 1.0
        # todo: check if normalized. If not, normalize it.
        # Q, T = SE3_to_quaternion_and_translation_torch(torch.from_numpy(R).unsqueeze(0).double())
        # assert torch.allclose(Q.float(), torch.from_numpy(qvec).float(), rtol=1e-3, atol=1e-3), (
        #     Q.float(), torch.from_numpy(qvec).float())
        # assert torch.allclose(T.float(), torch.from_numpy([px, py, pz]).float(), rtol=1e-3, atol=1e-3)
        # here the world coordinate is defined in robotics space, where z is up, x is left and y is right.
        # the camera coordinate is defined in opencv convention, where the camera is looking down the z axis,
        # y is down and x is right.

        # convert the world coordinate to camera coordinate.
        if output_convention == "opencv":
            R = S.T.dot(R)  # this is the c2w in opencv convention.

        world2cam = np.linalg.inv(R)  # this is the w2c in opencv convention.

        world2cam = torch.tensor(world2cam)
        Q, T = SE3_to_quaternion_and_translation_torch(world2cam.unsqueeze(0))  # this is the w2c
        qx, qy, qz, qw = Q.numpy().flatten().tolist()
        px, py, pz = T.numpy().flatten().tolist()

        qvec_w2c = np.array([qw, qx, qy, qz])
        assert np.linalg.norm(qvec_w2c) - 1 < 1e-3, np.linalg.norm(qvec_w2c)
        tvec_w2c = np.array([px, py, pz])

        # get the camera_id:
        camera_id = available_cam_list.index(cur_camera_name)

        images[image_id] = Image(id=image_id, qvec=qvec_w2c, tvec=tvec_w2c, camera_id=camera_id + 1,
                                 name=cur_name, xys=None, point3D_ids=None, )
    return cameras, images


import torch
from typing import Tuple


def rotation_matrix_to_quaternion_torch(
    R: torch.Tensor,  # (batch_size, 3, 3)
) -> torch.Tensor:
    q = torch.zeros(
        R.shape[0], 4, device=R.device, dtype=R.dtype
    )  # (batch_size, 4) x, y, z, w
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q0_mask = trace > 0
    q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
    q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
    q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
    if q0_mask.any():
        R_for_q0 = R[q0_mask]
        S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
        q[q0_mask, 3] = 0.25 / S_for_q0
        q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
        q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
        q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0

    if q1_mask.any():
        R_for_q1 = R[q1_mask]
        S_for_q1 = 2.0 * torch.sqrt(
            1 + R_for_q1[..., 0, 0] - R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2]
        )
        q[q1_mask, 0] = 0.25 * S_for_q1
        q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
        q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
        q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1

    if q2_mask.any():
        R_for_q2 = R[q2_mask]
        S_for_q2 = 2.0 * torch.sqrt(
            1 + R_for_q2[..., 1, 1] - R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2]
        )
        q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
        q[q2_mask, 1] = 0.25 * S_for_q2
        q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
        q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2

    if q3_mask.any():
        R_for_q3 = R[q3_mask]
        S_for_q3 = 2.0 * torch.sqrt(
            1 + R_for_q3[..., 2, 2] - R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1]
        )
        q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
        q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
        q[q3_mask, 2] = 0.25 * S_for_q3
        q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3
    return q


def SE3_to_quaternion_and_translation_torch(
    transform: torch.Tensor,  # (batch_size, 4, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    this gives x, y, z, w
    """

    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    q = rotation_matrix_to_quaternion_torch(R)
    return q, t


import sys


def readColmapCameras(
    cam_extrinsics: t.Dict[int, Image],
    cam_intrinsics: t.Dict[int, Colmap_Camera],
    *,
    images_folder: str,
    force_centered_pp: bool = False,
) -> t.List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))  # R for c2w
        T = np.array(extr.tvec)  # here the T is not -RT, T for w2c

        if intr.model == "SIMPLE_PINHOLE":
            raise RuntimeError("SIMPLE_PINHOLE cameras are not supported!")
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            if force_centered_pp:
                cx = width / 2
                cy = height / 2
            else:
                cx = intr.params[2]
                cy = intr.params[3]
        elif intr.model == "PINHOLE":
            focal_length_x = float(intr.params[0])
            focal_length_y = float(intr.params[1])
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            if force_centered_pp:
                cx = width / 2
                cy = height / 2
            else:
                cx = intr.params[2]
                cy = intr.params[3]
        else:
            assert False, (
                "Colmap camera model not handled: "
                "only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )
        import os
        image_path = os.path.join(images_folder, extr.name)
        image_name = "/".join(
            [Path(x).stem for x in Path(image_path).relative_to(images_folder).parts]
        )
        image = None

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            cx=cx,
            cy=cy,
            focal_y=focal_length_y,
            focal_x=focal_length_x,
            camera_extrinsic=intr.extrinsic,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


from jaxtyping import Float


@dataclass
class CameraInfo:
    uid: int
    """camera id in colmap"""
    R: np.ndarray
    """R for c2w """
    T: np.ndarray
    """T for w2c """
    FovY: float
    FovX: float
    image: np.ndarray | None
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float
    cy: float
    focal_x: float
    focal_y: float

    camera_extrinsic: np.ndarray | None = None

    @property
    def w2c(self) -> Float[np.ndarray, "4 4"]:
        return getWorld2View2(self.R, self.T)

    @property
    def c2w(self) -> Float[np.ndarray, "4 4"]:
        return np.linalg.inv(self.w2c)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(
    R, t, translate: np.ndarray | None = None, scale: float | None = None
) -> np.ndarray:
    """
    this gives the world2camera matrix which already takes the camera pose transformation,
    such as centering and scaling.
    return world to camera matrix of 4X4
    """
    if translate is None:
        translate = np.array([0.0, 0.0, 0.0])
    if scale is None:
        scale = 1.0

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # R for w2c
    Rt[:3, 3] = t  # T for w2c
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt.astype(np.float32)  # type: ignore
