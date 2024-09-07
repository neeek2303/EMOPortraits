import cv2
import numpy as np

from .gaze_tools import get_endpoint
from .utils import torch_image_to_numpy


def fix_expected_ptr_cv_umat(image):
    """
    Some OpenCV functions fail with
        TypeError: Expected Ptr<cv::UMat> for argument 'img'
    when they receive Numpy arrays that had been transposed (e.g. via torch.Tensor.permute()).
    The workaround is to make memory layout for such arrays contiguous.
    """
    return np.ascontiguousarray(image)


def visualize_original_image(image, subject):
    image = fix_expected_ptr_cv_umat(image)
    cv2.rectangle(
        image,
        (int(subject.box.left), int(subject.box.top)),
        (int(subject.box.right), int(subject.box.bottom)),
        (0, 0, 255), 2)

    for x, y in subject.landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0))

    if subject.headpose is not None:
        center_x = (subject.box.left + subject.box.right) / 2
        center_y = (subject.box.top + subject.box.bottom) / 2
        endpoint_x, endpoint_y = get_endpoint(subject.headpose.theta, subject.headpose.phi, center_x, center_y, 100)
        cv2.line(
            image,
            (int(center_x), int(center_y)),
            (int(endpoint_x), int(endpoint_y)),
            (0, 0, 255), 3)

    return image


def visualize_aligned_face(subject):
    aligned_face = fix_expected_ptr_cv_umat(torch_image_to_numpy(subject.aligned_face))
    cv2.rectangle(
        aligned_face,
        (int(subject.left_eye_bb.left), int(subject.left_eye_bb.top)),
        (int(subject.left_eye_bb.right), int(subject.left_eye_bb.bottom)),
        (0, 0, 255), 2)
    cv2.rectangle(
        aligned_face,
        (int(subject.right_eye_bb.left), int(subject.right_eye_bb.top)),
        (int(subject.right_eye_bb.right), int(subject.right_eye_bb.bottom)),
        (0, 0, 255), 2)
    for x, y in subject.transformed_eye_landmarks:
        cv2.circle(aligned_face, (int(x), int(y)), 2, (255, 0, 0))

    if subject.gaze is not None:
        for bb in (subject.left_eye_bb, subject.right_eye_bb):
            center_x = (bb.left + bb.right) / 2
            center_y = (bb.top + bb.bottom) / 2
            endpoint_x, endpoint_y = get_endpoint(subject.gaze.theta, subject.gaze.phi, center_x, center_y, 50)
            cv2.line(
                aligned_face,
                (int(center_x), int(center_y)),
                (int(endpoint_x), int(endpoint_y)),
                (0, 255, 0), 1)

    # Draw visualization of eyes at the center of the bottom edge of the image.
    gaze_vis = visualize_gaze(subject)
    dst_l = aligned_face.shape[1] // 2 - gaze_vis.shape[1] // 2
    dst_r = dst_l + gaze_vis.shape[1]
    src_l = 0 if dst_l >= 0 else -dst_l
    src_r = gaze_vis.shape[1] if dst_r <= aligned_face.shape[1] else gaze_vis.shape[1] - (dst_r - aligned_face.shape[1])
    dst_l = max(dst_l, 0)
    dst_r = min(dst_r, aligned_face.shape[1])
    dst_t = aligned_face.shape[0] - gaze_vis.shape[0]
    src_t = 0 if dst_t >= 0 else -dst_t
    dst_t = max(dst_t, 0)
    aligned_face[dst_t:aligned_face.shape[0], dst_l:dst_r] = gaze_vis[src_t:, src_l:src_r]

    return aligned_face


def visualize_gaze(subject):
    def visualize_eye_result(eye_image):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = fix_expected_ptr_cv_umat(torch_image_to_numpy(eye_image.detach()))

        if subject.gaze is not None:
            center_x = output_image.shape[1] / 2
            center_y = output_image.shape[0] / 2
            endpoint_x, endpoint_y = get_endpoint(subject.gaze.theta, subject.gaze.phi, center_x, center_y, 50)
            cv2.line(
                output_image,
                (int(center_x), int(center_y)),
                (int(endpoint_x), int(endpoint_y)),
                (0, 255, 0))

        return output_image

    r_gaze_img = visualize_eye_result(subject.right_eye_color)
    l_gaze_img = visualize_eye_result(subject.left_eye_color)
    s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

    return s_gaze_img
