import numpy as np
import pandas as pd
from PIL import Image
import torch




def procrustes(s1, s2):
    if len(s1.size()) < 3:
        s1 = s1.unsqueeze(0)
    if len(s2.size()) < 3:
        s2 = s2.unsqueeze(0)

    coordinates = s1.size(2)

    mu1 = s1.mean(axis=1, keepdims=True)
    mu2 = s2.mean(axis=1, keepdims=True)

    x1 = s1 - mu1
    x2 = s2 - mu2

    var1 = torch.sum(x1 ** 2, dim=1).sum(dim=1)

    cov = x1.transpose(1, 2).bmm(x2)
    u, s, v = torch.svd(cov.float())

    z = torch.eye(u.shape[1], device=s1.device).unsqueeze(0)
    z = z.repeat(u.shape[0], 1, 1)
    z[:, -1, -1] *= torch.sign(torch.det(u.bmm(v.transpose(1, 2)).float()))

    r = v.bmm(z.bmm(u.permute(0, 2, 1)))
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in r.bmm(cov)]) / var1
    t = mu2.view(-1, coordinates, 1) - (scale.unsqueeze(-1).unsqueeze(-1) * (r.bmm(mu1.view(-1, coordinates, 1))))

    return scale, r, t.squeeze()



def get_transform_matrix(rotation, translation, scale=None):

    batch_size = rotation.size(0)
    num_coordinates = rotation.size(1)

    trans = torch.zeros((batch_size, num_coordinates + 1, num_coordinates + 1), device=rotation.device)
    if scale is None:
        trans[:, :num_coordinates, :num_coordinates] = rotation
    else:
        trans[:, :num_coordinates, :num_coordinates] = scale.unsqueeze(-1).unsqueeze(-1) * rotation
    trans[:, :num_coordinates, num_coordinates] = translation.squeeze()
    trans[:, num_coordinates, num_coordinates] = 1

    return trans


def get_scale_matrix(rotation, translation, scale=None):

    batch_size = rotation.size(0)
    num_coordinates = rotation.size(1)

    trans = torch.zeros((batch_size, num_coordinates + 1, num_coordinates + 1), device=rotation.device)

    trans[:, :num_coordinates, :num_coordinates] = scale.unsqueeze(-1).unsqueeze(-1) * torch.eye(3).unsqueeze(0)
    trans[:, :num_coordinates, num_coordinates] = translation.squeeze()*0
    trans[:, num_coordinates, num_coordinates] = 1

    return trans




def transform_landmarks(ref, transformation):
    ret_np = False
    if isinstance(ref, np.ndarray):
        ret_np = True
        ref = torch.from_numpy(ref)
        transformation = torch.from_numpy(transformation)

    ref = ref.view(-1, ref.size(-2), ref.size(-1))
    transformation = transformation.view(-1, transformation.size(-3), transformation.size(-2), transformation.size(-1))

    seq_length = transformation.shape[1]
    no_points = ref.shape[-2]
    coordinates = ref.shape[-1]

    rot_matrix = transformation[:, :, :coordinates, :coordinates]
    out_translation = transformation[:, :, :coordinates, coordinates]

    out_landmarks = torch.bmm(ref[:, None, :, :].repeat(1, seq_length, 1, 1).view(-1, no_points, 3),
                              rot_matrix.view(-1, 3, 3).transpose(1, 2)).contiguous()

    out_landmarks = out_landmarks.view(-1, seq_length, no_points, coordinates) + out_translation[:, :, None, :]

    if ret_np:
        return out_landmarks.squeeze().numpy()
    else:
        return out_landmarks.squeeze()