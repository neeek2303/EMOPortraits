import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .estimate_gaze_pytorch import GazeEstimator
from .gaze_tools import (
    box_in_image,
    crop_face_from_image,
    FaceBox,
    get_square_box,
    move_box,
)
from .tracker_generic import TrackedSubject


logger: logging.Logger = logging.getLogger("rt_gene")


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def select_by_mask(a, mask):
    return [v for (is_true, v) in zip(mask, a) if is_true]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def expand_by_mask(mask, values):
    """Creates a new list with length equal to len(mask), whose values are None where mask is False
    and are taken from `values` otherwise. This implies that the number of true elements in the mask
    should be equal to len(values)."""
    a = [None] * len(mask)
    j = 0
    for i, is_true in enumerate(mask):
        if is_true:
            a[i] = values[j]
            j += 1
    return a


def get_faceboxes_from_batch(
    # pyre-fixme[2]: Parameter must be annotated.
    face_detector,
    image_batch: torch.Tensor,
    largest_only: bool = True,
) -> Union[List[Optional[FaceBox]], List[List[FaceBox]]]:
    fraction = min(
        4.0, image_batch.shape[2] / 32, image_batch.shape[3] / 32
    )  # SFD crashes on inputs smaller than 32px
    if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")) >= (1, 5, 0):
        image_batch = torch.nn.functional.interpolate(
            image_batch,
            scale_factor=1 / fraction,
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=False,
        )
    else:
        image_batch = torch.nn.functional.interpolate(
            image_batch, scale_factor=1 / fraction, mode="bicubic", align_corners=False
        )

    # It is unclear which color channel order is expected by the face detector.
    # On one hand, get_landmarks_from_image() does RGB->BGR conversion before
    # feeding images into the face detector. On the other hand, SFDDetector loads
    # images via tensor_or_path_to_ndarray(), which returns RGB images by default.
    # In any case, the detector seems to work either way.
    detections = face_detector.detect_from_batch(image_batch * 255)

    result = []
    for image_detections in detections:
        image_faceboxes = []
        for detection in image_detections:
            box = detection[:4]
            confidence = detection[4]

            if box_in_image(box, image_batch[0]) and confidence > 0.6:
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = move_box(box, [0, offset_y])

                # Make box square.
                facebox = get_square_box(box_moved)
                image_faceboxes.append(FaceBox(*facebox))

        if largest_only:
            # Select largest facebox by area, if there is at least one detection.
            if image_faceboxes:
                facebox = max(
                    image_faceboxes,
                    key=lambda box: (box.right - box.left) * (box.bottom - box.top),
                )
            else:
                facebox = None
            result.append(facebox)
        else:
            # Sort faceboxes by area in descending order.
            image_faceboxes.sort(
                key=lambda box: (box.right - box.left) * (box.bottom - box.top),
                reverse=True,
            )
            result.append(image_faceboxes)

    return result


class RtGeneEstimator:
    # This is intentionally not an nn.Module subclass. We want to:
    # 1) Use it as a criterion, on GPU;
    # 2) Use it as a metric, on CPU;
    # 2) Store it in training_module.
    # training_module is wrapped in DistributedDataParallel, which crashes with an error on initialization if
    # the module it wraps refers to any CPU modules.

    EYE_IMAGE_SIZE = (60, 36)

    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        device,
        model_nets_path: List[str],
        # pyre-fixme[2]: Parameter must be annotated.
        gaze_model_types=("vgg16",),
        interpolate: bool = False,
    ) -> None:
        """
        gaze_model_ids: can be any subset of [1, 2, 3, 4, 5, 6]
        """
        logger.debug("Loading gaze estimator...")
        self.gaze_estimator: GazeEstimator = GazeEstimator(
            model_nets_path, gaze_model_types, interpolate=interpolate, device=device
        ).to(device)
        logger.debug("... gaze estimator loaded.")

    def _extract_eye_patches(
        self,
        image_batch: torch.Tensor,
        faceboxes: List[FaceBox],
        landmarks: List[np.ndarray],
    ) -> List[TrackedSubject]:
        assert isinstance(image_batch, torch.Tensor)

        subjects = [
            TrackedSubject(
                box=box,
                face=crop_face_from_image(image, box),
                landmarks=l,
            )
            for box, image, l in zip(faceboxes, image_batch, landmarks)
        ]
        for subject in subjects:
            if subject is not None:
                subject.compute_eye_images_from_landmarks(self.EYE_IMAGE_SIZE)

        return subjects

    def get_eye_embeddings(
        self,
        image_batch: torch.Tensor,
        layer_indices: Tuple[int],
        faceboxes: List[FaceBox],
        landmarks: List[np.ndarray],
    ) -> List[TrackedSubject]:
        b, c, h, w = image_batch.shape
        assert c == 3

        subjects = self._extract_eye_patches(image_batch, faceboxes, landmarks)

        # Step 4: run eye patches through gaze estimation models to obtain eye embeddings suitable for optimization
        #   Input: image patches for both eyes of each of the detected subjects (currently at most 1 per batch element)
        #   Output: a list of tensors, BxMxCx... each, where M = number of gaze estimation models and
        #   Cx... are different for different tensors. The length of the list is equal to the number
        #   of layers where we compute the loss.
        has_eyes = [subject is not None and subject.has_eyes for subject in subjects]
        subjects_for_embeddings = select_by_mask(subjects, has_eyes)
        if subjects_for_embeddings:
            eye_embeddings = self.gaze_estimator.get_eye_embeddings(
                subjects_for_embeddings, layer_indices
            )
            eye_embeddings_per_subject_with_eyes = [
                [eye_embeddings[j][i] for j in range(len(eye_embeddings))]
                for i in range(len(eye_embeddings[0]))
            ]
            eye_embeddings_per_subject = expand_by_mask(
                has_eyes, eye_embeddings_per_subject_with_eyes
            )
            for has, subject, emb in zip(
                has_eyes, subjects, eye_embeddings_per_subject
            ):
                if has:
                    subject.eye_embeddings = emb

        return subjects


class RtGeneTrainer(nn.Module, RtGeneEstimator):
    def __init__(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        device,
        model_nets_path: List[str],
        # pyre-fixme[2]: Parameter must be annotated.
        gaze_model_types=("vgg16",),
        interpolate: bool = False,
    ) -> None:
        """
        gaze_model_ids: can be any subset of [1, 2, 3, 4, 5, 6]
        """
        super().__init__()
        logger.debug("Loading gaze estimator...")
        self.gaze_estimator: GazeEstimator = GazeEstimator(
            model_nets_path, gaze_model_types, interpolate=interpolate, device=device
        ).to(device)
        logger.debug("... gaze estimator loaded.")
