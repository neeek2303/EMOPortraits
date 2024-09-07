"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

import cv2
import numpy as np
import torch.nn.functional as F

from .gaze_tools import FaceBox, get_normalised_eye_landmarks, torch_warpaffine


class TrackedSubject(object):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, box, face, landmarks) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.box = box
        # pyre-fixme[4]: Attribute must be annotated.
        self.face_color = face
        # pyre-fixme[4]: Attribute must be annotated.
        self.landmarks = landmarks

        # pyre-fixme[4]: Attribute must be annotated.
        self.aligned_face = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.transformed_eye_landmarks = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.left_eye_color = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.right_eye_color = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.left_eye_bb = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.right_eye_bb = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.headpose = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.gaze = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.eye_embeddings = None
        # pyre-fixme[4]: Attribute must be annotated.
        self.kp2d = None

    def __repr__(self) -> str:
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def repr_array(array):
            if array is None:
                return "None"
            else:
                a_type = "ndarray" if isinstance(array, np.ndarray) else "tensor"
                return f"<{array.shape} {array.dtype} {a_type}>"

        params = [
            f"box={self.box}",
            f"face={repr_array(self.face_color)}",
            f"landmarks={repr_array(self.landmarks)}",
            f"aligned_face={repr_array(self.aligned_face)}",
            f"transformed_eye_landmarks={repr_array(self.transformed_eye_landmarks)}",
            f"left_eye_color={repr_array(self.left_eye_color)}",
            f"right_eye_color={repr_array(self.right_eye_color)}",
            f"left_eye_bb={self.left_eye_bb}",
            f"right_eye_bb={self.right_eye_bb}",
            f"headpose={self.headpose}",
            f"gaze={self.gaze}",
            f"eye_embeddings={repr_array(self.eye_embeddings)}",
        ]
        params_str = ",\n".join("    " + p for p in params)
        return f"{self.__class__.__name__}(\n{params_str})"

    @property
    def has_eyes(self) -> bool:
        return self.left_eye_color is not None and self.right_eye_color is not None

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def compute_distance(self, other_element):
        return np.sqrt(np.sum((self.box - other_element.box) ** 2))

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def compute_eye_images_from_landmarks(self, eye_image_size):
        eye_landmarks = get_normalised_eye_landmarks(self.landmarks, self.box)
        margin_ratio = 1.0
        desired_ratio = float(eye_image_size[1]) / float(eye_image_size[0]) / 2.0

        # Get the width of the eye, and compute how big the margin should be according to the width
        lefteye_width = eye_landmarks[3][0] - eye_landmarks[2][0]
        righteye_width = eye_landmarks[1][0] - eye_landmarks[0][0]

        lefteye_center_x = eye_landmarks[2][0] + lefteye_width / 2
        righteye_center_x = eye_landmarks[0][0] + righteye_width / 2
        lefteye_center_y = (eye_landmarks[2][1] + eye_landmarks[3][1]) / 2.0
        righteye_center_y = (eye_landmarks[1][1] + eye_landmarks[0][1]) / 2.0

        aligned_face, rot_matrix = self.align_face_to_eyes(
            self.face_color,
            right_eye_center=(righteye_center_x, righteye_center_y),
            left_eye_center=(lefteye_center_x, lefteye_center_y),
        )

        # rotate the eye landmarks by same affine rotation to extract the correct landmarks
        ones = np.ones(shape=(len(eye_landmarks), 1))
        points_ones = np.hstack([eye_landmarks, ones])
        transformed_eye_landmarks = rot_matrix.dot(points_ones.T).T

        self.aligned_face = aligned_face
        self.transformed_eye_landmarks = transformed_eye_landmarks

        # recompute widths, margins and centers
        lefteye_width = (
            transformed_eye_landmarks[3][0] - transformed_eye_landmarks[2][0]
        )
        righteye_width = (
            transformed_eye_landmarks[1][0] - transformed_eye_landmarks[0][0]
        )
        lefteye_margin, righteye_margin = (
            lefteye_width * margin_ratio,
            righteye_width * margin_ratio,
        )
        lefteye_center_y = (
            transformed_eye_landmarks[2][1] + transformed_eye_landmarks[3][1]
        ) / 2.0
        righteye_center_y = (
            transformed_eye_landmarks[1][1] + transformed_eye_landmarks[0][1]
        ) / 2.0

        # Now compute the bounding boxes
        # The left / right x-coordinates are computed as the landmark position plus/minus the margin
        # The bottom / top y-coordinates are computed according to the desired ratio, as the width of the image is known
        left_bb = np.zeros(4, dtype=int)
        left_bb[0] = transformed_eye_landmarks[2][0] - lefteye_margin / 2.0
        left_bb[1] = lefteye_center_y - (lefteye_width + lefteye_margin) * desired_ratio
        left_bb[2] = transformed_eye_landmarks[3][0] + lefteye_margin / 2.0
        left_bb[3] = lefteye_center_y + (lefteye_width + lefteye_margin) * desired_ratio
        left_bb = FaceBox(*left_bb)

        right_bb = np.zeros(4, dtype=int)
        right_bb[0] = transformed_eye_landmarks[0][0] - righteye_margin / 2.0
        right_bb[1] = (
            righteye_center_y - (righteye_width + righteye_margin) * desired_ratio
        )
        right_bb[2] = transformed_eye_landmarks[1][0] + righteye_margin / 2.0
        right_bb[3] = (
            righteye_center_y + (righteye_width + righteye_margin) * desired_ratio
        )
        right_bb = FaceBox(*right_bb)

        # Extract the eye images from the aligned image
        left_eye_color = aligned_face[
            :, left_bb[1] : left_bb[3], left_bb[0] : left_bb[2]
        ]
        right_eye_color = aligned_face[
            :, right_bb[1] : right_bb[3], right_bb[0] : right_bb[2]
        ]

        if 0 in left_eye_color.shape[-2:] or 0 in right_eye_color.shape[-2:]:
            return None, None, None, None

        # So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
        left_eye_color_resized = F.interpolate(
            left_eye_color.unsqueeze(0),
            eye_image_size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
        right_eye_color_resized = F.interpolate(
            right_eye_color.unsqueeze(0),
            eye_image_size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        self.left_eye_color = left_eye_color_resized
        self.right_eye_color = right_eye_color_resized
        self.left_eye_bb = left_bb
        self.right_eye_bb = right_bb

    @staticmethod
    # pyre-fixme[3]: Return type must be annotated.
    def align_face_to_eyes(
        # pyre-fixme[2]: Parameter must be annotated.
        face_img,
        # pyre-fixme[2]: Parameter must be annotated.
        right_eye_center,
        # pyre-fixme[2]: Parameter must be annotated.
        left_eye_center,
        # pyre-fixme[2]: Parameter must be annotated.
        face_width=None,
        # pyre-fixme[2]: Parameter must be annotated.
        face_height=None,
    ):
        # modified from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
        c, h, w = face_img.shape
        assert c == 3, c

        desired_left_eye = (0.35, 0.35)
        desired_face_width = face_width if face_width is not None else w
        desired_face_height = face_height if face_height is not None else h
        #         print(desired_face_width,desired_face_width)
        # compute the angle between the eye centroids
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x)).item() - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((d_x**2) + (d_y**2))
        desired_dist = desired_right_eye_x - desired_left_eye[0]
        desired_dist *= desired_face_width
        scale = desired_dist / max(
            dist.item(), 1
        )  # max(..., 1) is a workaround for division by zero

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = (
            (left_eye_center[0] + right_eye_center[0]) // 2,
            (left_eye_center[1] + right_eye_center[1]) // 2,
        )

        # grab the rotation matrix for rotating and scaling the face
        m = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        t_x = desired_face_width * 0.5
        t_y = desired_face_height * desired_left_eye[1]
        m[0, 2] += t_x - eyes_center[0]
        m[1, 2] += t_y - eyes_center[1]

        # apply the affine transformation
        (w, h) = (desired_face_width, desired_face_height)
        aligned_face = torch_warpaffine(face_img, m, (w, h), flags=cv2.INTER_LINEAR)
        return aligned_face, m
