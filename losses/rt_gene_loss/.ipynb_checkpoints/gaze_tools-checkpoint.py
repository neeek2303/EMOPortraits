"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

import math

import numpy as np


def get_phi_theta_from_euler(euler_angles):
    return -euler_angles[2], -euler_angles[1]


def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y


def limit_yaw(euler_angles_head):
    # [0]: pos - roll right, neg -   roll left
    # [1]: pos - look down,  neg -   look up
    # [2]: pos - rotate left,  neg - rotate right
    euler_angles_head[2] += np.pi
    if euler_angles_head[2] > np.pi:
        euler_angles_head[2] -= 2 * np.pi

    return euler_angles_head


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
    return image_torch[:, _bb[1]: _bb[3], _bb[0]: _bb[2]]


def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]

    return [left_x, top_y, right_x, bottom_y]


def box_in_image(box, image):
    """Check if the box is in image"""
    if isinstance(image, np.ndarray):
        h, w, c = image.shape
    else:
        c, h, w = image.shape
    assert c == 3

    return box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h


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


def get_normalised_eye_landmarks(landmarks, box):
    eye_indices = np.array([36, 39, 42, 45])
    transformed_landmarks = landmarks[eye_indices]
    transformed_landmarks[:, 0] -= box[0]
    transformed_landmarks[:, 1] -= box[1]
    return transformed_landmarks
