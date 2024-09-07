"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
import math
from collections import namedtuple
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# pyre-fixme[4]: Attribute annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
FaceBox = namedtuple("FaceBox", ["left", "top", "right", "bottom"])


# pyre-fixme[2]: Parameter must be annotated.
def freeze_model(model) -> None:
    model.eval()
    model.requires_grad_(False)

    # Forbid doing .train(), .eval() and .parameters()
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def train_noop(self, mode=True):
        pass

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def parameters_noop(self, recurse=True):
        return []

    model.train = train_noop.__get__(model, nn.Module)
    model.parameters = parameters_noop.__get__(model, nn.Module)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def torch_image_to_numpy(image_torch, inplace: bool = False, contiguous: bool = False):
    """Convert PyTorch tensor to Numpy array.
    :param image_torch: PyTorch float CHW Tensor in range [0..1].
    :param inplace: modify the tensor in-place.
    :returns: Numpy uint8 HWC array in range [0..255]."""
    assert len(image_torch.shape) == 3
    assert image_torch.shape[0] == 3
    if not inplace:
        image_torch = image_torch.clone()
    result = (
        image_torch.mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    if contiguous:
        result = np.ascontiguousarray(result)
    return result


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def torch_warpaffine(img: torch.Tensor, M: np.ndarray, dsize, flags=cv2.INTER_NEAREST):
    # Source: https://github.com/wuneng/WarpAffine2GridSample/tree/125ed9bd2b26a5c2e4567ec9a39f9309b33e7521

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_N(W, H):
        """N that maps from unnormalized to normalized coordinates"""
        N = np.zeros((3, 3), dtype=np.float64)
        N[0, 0] = 2.0 / W
        N[0, 1] = 0
        N[1, 1] = 2.0 / H
        N[1, 0] = 0
        N[0, -1] = -1.0
        N[1, -1] = -1.0
        N[-1, -1] = 1.0
        return N

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_N_inv(W, H):
        """N that maps from normalized to unnormalized coordinates"""
        # TODO: do this analytically maybe?
        N = get_N(W, H)
        return np.linalg.inv(N)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def convert_M_to_theta(M, w, h):
        """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
        compatible with `torch.F.affine_grid`
        Note:
        M works with `opencv.warpAffine`.
        To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required
        Parameters
        ----------
        M : np.ndarray
            affine warp matrix shaped [2, 3]
        w : int
            width of image
        h : int
            height of image
        Returns
        -------
        np.ndarray
            theta tensor for `torch.F.affine_grid`, shaped [2, 3]
        """
        M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
        M_aug[-1, -1] = 1.0
        N = get_N(w, h)
        N_inv = get_N_inv(w, h)
        theta = N @ M_aug @ N_inv
        theta = np.linalg.inv(theta)
        return theta[:2, :]

    w, h = dsize
    theta = convert_M_to_theta(M, w, h)
    grid = F.affine_grid(
        torch.tensor(theta, dtype=torch.float32, device=img.device).unsqueeze(0),
        [1, 3, h, w],
        align_corners=False,
    )
    mode = {
        cv2.INTER_NEAREST: "nearest",
        cv2.INTER_LINEAR: "bilinear",
    }[flags]
    result = F.grid_sample(
        img.unsqueeze(0), grid, mode=mode, align_corners=False
    ).squeeze(0)
    return result


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def crop_face_from_image(image_torch, box):
    c, h, w = image_torch.shape
    assert c == 3, c
    _bb = list(map(int, box))
    if _bb[0] < 0:
        _bb[0] = 0
    if _bb[1] < 0:
        _bb[1] = 0
    if _bb[2] > w:
        _bb[2] = w
    if _bb[3] > h:
        _bb[3] = h
    return image_torch[:, _bb[1] : _bb[3], _bb[0] : _bb[2]]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]

    return [left_x, top_y, right_x, bottom_y]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def box_in_image(box, image):
    """Check if the box is in image"""
    if isinstance(image, np.ndarray):
        h, w, c = image.shape
    else:
        c, h, w = image.shape
    assert c == 3

    return box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    return [left_x, top_y, right_x, bottom_y]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_normalised_eye_landmarks(landmarks, box):
    eye_indices = np.array([36, 39, 42, 45])
    transformed_landmarks = landmarks[eye_indices]
    transformed_landmarks[:, 0] -= box[0]
    transformed_landmarks[:, 1] -= box[1]
    return transformed_landmarks


def get_phi_theta_from_euler(euler_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return -euler_angles[2], -euler_angles[1]


def get_euler_from_phi_theta(phi: float, theta: float) -> Tuple[float, float, float]:
    return 0, -theta, -phi


def limit_yaw(euler_angles_head: np.ndarray) -> np.ndarray:
    # [0]: pos - roll right, neg -   roll left
    # [1]: pos - look down,  neg -   look up
    # [2]: pos - rotate left,  neg - rotate right
    euler_angles_head[2] += np.pi
    if euler_angles_head[2] > np.pi:
        euler_angles_head[2] -= 2 * np.pi

    return euler_angles_head


# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}


def euler_from_matrix(
    matrix: np.ndarray, axes: str = "sxyz"
) -> Tuple[float, float, float]:

    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az
