# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from typing import List

import numpy as np

import torch
import torch.nn.functional as F
from .rt_gaze_estimation_models import (
    GazeEstimationModelVGG,
)
from .tracker_generic import TrackedSubject

from torch import nn
from torchvision import transforms

from .gaze_estimation_models_pytorch import GazeEstimationModel
from .gaze_tools import freeze_model


class GazeEstimator(nn.Module):
    def __init__(
        self,
        model_files: List[str],
        model_types: List[str],
        interpolate: bool,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        if not isinstance(model_files, list):
            model_files = [model_files]

        if not isinstance(model_types, list):
            model_types = [model_types]

        self._transform: transforms.Compose = transforms.Compose([])
        if interpolate:
            self._transform = transforms.Compose(
                [
                    lambda x: F.interpolate(
                        x.unsqueeze(0), (224, 224), mode="bicubic", align_corners=False
                    ).squeeze(0)
                ]
            )

        self._models: List[nn.Module] = []
        if len(model_files) == 0:
            # This is used to load an untrained model for distillation
            _model = GazeEstimationModel(model_type=model_types[0], device=device)
            self._model: GazeEstimationModel = _model
            self._models.append(_model)
        for i, (file, model_type) in enumerate(zip(model_files, model_types)):
            _model = GazeEstimationModel(model_type=model_type, device=device)
            _model.load_state_dict(torch.load(file, map_location="cpu"))

            freeze_model(_model)

            setattr(self, f"_model_{i}", _model)
            self._models.append(_model)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _get_transformed_inputs(self, subjects):
        left_eyes = torch.stack(
            # pyre-ignore
            [self._transform(subject.left_eye_color) for subject in subjects]
        )
        right_eyes = torch.stack(
            # pyre-ignore
            [self._transform(subject.right_eye_color) for subject in subjects]
        )

        return left_eyes, right_eyes

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_eye_embeddings(self, subjects, layer_indices):
        left_eyes, right_eyes = self._get_transformed_inputs(subjects)

        # List[Tuple[list for left eye, list for right eye]]
        eye_embeddings_per_model = [
            model.get_eye_embeddings(left_eyes, right_eyes, layer_indices)
            for model in self._models
        ]

        left_eyes_flip = torch.flip(left_eyes, [3])
        right_eyes_flip = torch.flip(right_eyes, [3])
        eye_embeddings_per_model_flip = [
            model.get_eye_embeddings(right_eyes_flip, left_eyes_flip, layer_indices)
            for model in self._models
        ]

        eye_embeddings = []
        for i in range(
            len(eye_embeddings_per_model[0][0])
        ):  # Iterate over embeddings from different layers
            layer_embeddings = []
            # Iterate over embeddings from different models
            for (left_emb, right_emb), (right_emb_flip, left_emb_flip) in zip(
                eye_embeddings_per_model, eye_embeddings_per_model_flip
            ):
                # Concat left & right eye embeddings from specific layer along channel axis
                layer_embeddings.append(
                    torch.cat(
                        [
                            left_emb[i],
                            left_emb_flip[i],
                            right_emb[i],
                            right_emb_flip[i],
                        ],
                        dim=1,
                    )
                )
            # Stack embeddings from different models along a new axis (new tensor shape is BxMxCx..., where M=models)
            layer_embeddings = torch.stack(layer_embeddings, dim=1)
            assert layer_embeddings.shape[1] == len(self._models), (
                layer_embeddings.shape[1],
                len(self._models),
            )
            eye_embeddings.append(layer_embeddings)

        return eye_embeddings

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def estimate_gaze_twoeyes(self, subjects, mode: str = "sep"):
        assert mode in ["sep", "mean"]
        assert subjects[0].kp2d is not None, "Don't have kp2d"
        left_eyes, right_eyes = self._get_transformed_inputs(subjects)
        kp2d = torch.as_tensor(
            [subject.kp2d for subject in subjects], device=left_eyes.device
        )
        result = [model(left_eyes, right_eyes, kp2d) for model in self._models]
        if mode == "sep":
            return result
        elif mode == "mean":
            result = torch.stack(result, dim=0).mean(dim=0)
            return result


class RTGazeEstimator(nn.Module):
    """
    NOTE: this model can be used only for research and testing. It CANNOT be used in prod.
    """

    def __init__(self, device: torch.device, model_files: List[str]) -> None:
        super(RTGazeEstimator, self).__init__()
        self.model_files = model_files
        self.device = device
        if not isinstance(model_files, list):
            self.model_files = [model_files]

        if len(self.model_files) == 1:
            self._gaze_offset: float = 0.11
        else:
            self._gaze_offset = 0.0

        self._models: List[nn.Module] = []
        for ckpt in self.model_files:
            try:
                _model = GazeEstimationModelVGG(num_out=2)
                _model.load_state_dict(torch.load(ckpt))
                _model.to(self.device)
                _model.eval()
                self._models.append(_model)
            except Exception as e:
                print("Error loading checkpoint", ckpt)
                raise e

    def estimate_gaze_twoeyes(
        self,
        subjects: List[TrackedSubject],
        inference_headpose_list: List[torch.Tensor],
    ) -> np.ndarray:
        left_eyes = torch.stack([subject.left_eye_color for subject in subjects]).to(
            self.device
        )
        right_eyes = torch.stack([subject.right_eye_color for subject in subjects]).to(
            self.device
        )
        tranformed_head = torch.as_tensor(inference_headpose_list).to(self.device)
        result = [
            model(left_eyes, right_eyes, tranformed_head).detach().cpu()
            for model in self._models
        ]
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1).numpy()
        result[:, 1] += self._gaze_offset
        return result
