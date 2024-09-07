import logging
from pathlib import Path
from typing import Tuple, List, Optional, Union

import cv2
import face_alignment
import numpy as np
import torch
import torchvision
from .estimate_gaze_pytorch import GazeEstimator
from .gaze_tools import get_phi_theta_from_euler, limit_yaw, get_square_box
from .gaze_tools_standalone import euler_from_matrix
from .tracker_generic import TrackedSubject
from .utils import torch_image_to_numpy, FaceBox, SphericalVector


logger = logging.getLogger('rt_gene')


def select_by_mask(a, mask):
    return [v for (is_true, v) in zip(mask, a) if is_true]


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


def get_faceboxes_from_batch(face_detector,
                             image_batch: torch.Tensor,
                             largest_only=True) -> Union[List[Optional[FaceBox]], List[List[FaceBox]]]:
    fraction = min(4.0, image_batch.shape[2] / 32, image_batch.shape[3] / 32)  # SFD crashes on inputs smaller than 32px
    if tuple(int(v) for v in torch.__version__.split('+')[0].split('.')) >= (1, 5, 0):
        image_batch = torch.nn.functional.interpolate(
            image_batch, scale_factor=1 / fraction, mode='bicubic', align_corners=False, recompute_scale_factor=False)
    else:
        image_batch = torch.nn.functional.interpolate(
            image_batch, scale_factor=1 / fraction, mode='bicubic', align_corners=False)

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

            if gaze_tools.box_in_image(box, image_batch[0]) and confidence > 0.6:
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = gaze_tools.move_box(box, [0, offset_y])

                # Make box square.
                facebox = gaze_tools.get_square_box(box_moved)
                image_faceboxes.append(FaceBox(*facebox))

        if largest_only:
            # Select largest facebox by area, if there is at least one detection.
            if image_faceboxes:
                facebox = max(image_faceboxes, key=lambda box: (box.right - box.left) * (box.bottom - box.top))
            else:
                facebox = None
            result.append(facebox)
        else:
            # Sort faceboxes by area in descending order.
            image_faceboxes.sort(key=lambda box: (box.right - box.left) * (box.bottom - box.top), reverse=True)
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

    def __init__(self, device, model_nets_path: List[str], gaze_model_types=('vgg16',), interpolate=False, align_face=True):
        """
        gaze_model_ids: can be any subset of [1, 2, 3, 4, 5, 6]
        """
        super().__init__()

        if align_face:
            logger.debug('Loading face alignment model...')
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
            logger.debug('... face alignment model loaded.')
        else:
            self.fa = None



        logger.debug('Loading gaze estimator...')
        self.gaze_estimator = GazeEstimator(model_nets_path, gaze_model_types, interpolate=interpolate, device=device).to(device)
        logger.debug('... gaze estimator loaded.')

    def _extract_eye_patches(
            self,
            image_batch: torch.Tensor,
            faceboxes: Optional[List[FaceBox]] = None,
            landmarks: Optional[List[np.ndarray]] = None) -> List[Optional[TrackedSubject]]:
        assert isinstance(image_batch, torch.Tensor)

        # Step 1: face detection. Finds the largest face bounding box in the image.
        #   Input: torch.Tensor in range [0..1] with shape BxCxHxW.
        #   Output: List[Optional[Facebox]]
        if faceboxes is None:
            if self.fa is None:
                raise ValueError('extract_eye_patches() did not receive faceboxes, but face alignment is not loaded')
            if landmarks is not None:
                raise ValueError('extract_eye_patches() received landmarks but not faceboxes?')
            image_batch = image_batch.cpu()
            with torch.no_grad():
                faceboxes: List[Optional[FaceBox]] = get_faceboxes_from_batch(
                    self.fa.face_detector, image_batch, largest_only=True)
        has_facebox = [facebox is not None for facebox in faceboxes]
        if not any(has_facebox):
            return [None] * len(image_batch)

        # Step 2: landmark estimation. Necessary for extracting eye patches.
        #   Input: torch.Tensor in range [0..1] with shape BxCxHxW; faceboxes detected earlier.
        #   Output: a list of tuples (facebox, face_image, landmarks).
        images_with_face = image_batch[has_facebox]
        if landmarks is None:
            # Note: since we left only the largest facebox in Step 1, we need to pass facebox in a list and then
            # extract the first (the only) element from the result.
            faceboxes_for_landmarks = [[facebox] for facebox in select_by_mask(faceboxes, has_facebox)]
            with torch.no_grad():
                landmarks_ = self.fa.get_landmarks_from_batch(images_with_face * 255, faceboxes_for_landmarks)
            landmarks_ = [l[0] for l in landmarks_]
            # The following is a workaround for something like `landmarks[has_facebox] = landmarks_`,
            # which I couldn't get to work with Numpy.
            landmarks: List[Optional[np.ndarray]] = expand_by_mask(has_facebox, landmarks_)

        subjects = [
            TrackedSubject(
                box=box,
                face=gaze_tools.crop_face_from_image(image, box),
                landmarks=l,
            ) if has else None
            for (has, box, image, l) in zip(has_facebox, faceboxes, image_batch, landmarks)
        ]
#         print(image_batch[0], faceboxes[0], gaze_tools.crop_face_from_image(image_batch[0], faceboxes[0]))
        # Step 3: extract eye image patches.
        #   Input: tuple (facebox, face_image, landmarks).
        #   Output: image patches and bounding boxes for the two eyes.
        # Populates subject.{left,right}_eye_color.
        for subject in subjects:
            if subject is not None:
                subject.compute_eye_images_from_landmarks(self.EYE_IMAGE_SIZE)

        return subjects

    def get_eye_embeddings(
            self,
            image_batch: torch.Tensor,
            layer_indices: Tuple[int],
            faceboxes: Optional[FaceBox] = None, landmarks: Optional[np.ndarray] = None) -> List[Optional[TrackedSubject]]:
        b, c, h, w = image_batch.shape
        assert c == 3

        subjects = self._extract_eye_patches(image_batch, faceboxes, landmarks)

        # Step 4: run eye patches through gaze estimation models to obtain eye embeddings suitable for optimization
        #   Input: image patches for both eyes of each of the detected subjects (currently at most 1 per batch element)
        #   Output: a list of tensors, BxMxCx... each, where M = number of gaze estimation models and
        #   Cx... are different for different tensors. The length of the list is equal to the number
        #   of layers where we compute the loss.
        has_eyes = [ #проверка на то, есть ли картинка для каждого глаза TODO добавить условие что человек не моргает
            subject is not None and
            subject.has_eyes
            for subject in subjects
        ]
        subjects_for_embeddings = select_by_mask(subjects, has_eyes)
        if subjects_for_embeddings:
            eye_embeddings = self.gaze_estimator.get_eye_embeddings(subjects_for_embeddings, layer_indices)
            eye_embeddings_per_subject_with_eyes = [
                [eye_embeddings[j][i] for j in range(len(eye_embeddings))]
                for i in range(len(eye_embeddings[0]))
            ]
            eye_embeddings_per_subject = expand_by_mask(has_eyes, eye_embeddings_per_subject_with_eyes)
            for has, subject, emb in zip(has_eyes, subjects, eye_embeddings_per_subject):
                if has:
                    subject.eye_embeddings = emb

        return subjects
