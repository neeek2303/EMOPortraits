from collections import namedtuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FaceBox = namedtuple('FaceBox', ['left', 'top', 'right', 'bottom'])
SphericalVector = namedtuple('SphericalVector', ['theta', 'phi'])


def freeze_model(model):
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # Forbid doing .train(), .eval() and .parameters()
    def train_noop(self, mode=True): pass
    def parameters_noop(self, recurse=True): return []

    model.train = train_noop.__get__(model, nn.Module)
    model.parameters = parameters_noop.__get__(model, nn.Module)


def torch_image_to_numpy(image_torch, inplace=False, contiguous=False):
    """Convert PyTorch tensor to Numpy array.
    :param image_torch: PyTorch float CHW Tensor in range [0..1].
    :param inplace: modify the tensor in-place.
    :returns: Numpy uint8 HWC array in range [0..255]."""
    assert len(image_torch.shape) == 3
    assert image_torch.shape[0] == 3
    if not inplace:
        image_torch = image_torch.clone()
    result = image_torch.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if contiguous:
        result = np.ascontiguousarray(result)
    return result


def torch_warpaffine(img: torch.Tensor, M: np.ndarray, dsize, flags=cv2.INTER_NEAREST):
    # Source: https://github.com/wuneng/WarpAffine2GridSample/tree/125ed9bd2b26a5c2e4567ec9a39f9309b33e7521

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

    def get_N_inv(W, H):
        """N that maps from normalized to unnormalized coordinates"""
        # TODO: do this analytically maybe?
        N = get_N(W, H)
        return np.linalg.inv(N)

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
        cv2.INTER_NEAREST: 'nearest',
        cv2.INTER_LINEAR: 'bilinear',
    }[flags]
    result = F.grid_sample(img.unsqueeze(0), grid, mode=mode, align_corners=False).squeeze(0)
    return result
