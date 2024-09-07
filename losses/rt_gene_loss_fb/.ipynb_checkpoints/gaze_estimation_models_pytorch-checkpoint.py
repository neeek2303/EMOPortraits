from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models


class KeypointsEncoder(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_harmonics: int,
        num_channels: int,
        num_layers: int,
        output_channels: int,
        output_size: int,
    ) -> None:
        super(KeypointsEncoder, self).__init__()
        self.output_channels = output_channels
        self.output_size = output_size

        self.register_buffer(
            "frequency_bands",
            2.0
            ** torch.linspace(0.0, num_harmonics - 1, num_harmonics).view(
                1, 1, 1, num_harmonics
            ),
            persistent=False,
        )

        layers_ = [nn.Linear(num_inputs * (2 + 2 * 2 * num_harmonics), num_channels)]

        for _ in range(max(num_layers - 2, 0)):
            # pyre-fixme[6]: For 1st argument expected `Iterable[Linear]` but got
            #  `Iterable[Union[ReLU, Linear]]`.
            layers_ += [nn.ReLU(inplace=True), nn.Linear(num_channels, num_channels)]

        # pyre-fixme[6]: For 1st argument expected `Iterable[Linear]` but got
        #  `Iterable[Union[ReLU, Linear]]`.
        layers_ += [
            nn.ReLU(inplace=True),
            #             nn.Linear(num_channels, output_channels * output_size**2, bias=False)]
            nn.Linear(num_channels, output_channels, bias=False),
        ]

        self.net = nn.Sequential(*layers_)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, kp):
        kp = kp[..., None]

        # Harmonic encoding: B x 68 x 2 x 1 + 2 * num_harmonics
        z = torch.cat(
            [
                kp,
                torch.sin(kp * self.frequency_bands),
                torch.cos(kp * self.frequency_bands),
            ],
            dim=3,
        )
        z = z.view(z.shape[0], -1)

        z = self.net(z)
        #         z = z.view(z.shape[0], self.output_channels, self.output_size, self.output_size)
        z = z.view(z.shape[0], self.output_channels)
        return z


class EyesEncoderVGG(nn.Module):
    def __init__(self, eyes_out: int = 128) -> None:
        super(EyesEncoderVGG, self).__init__()
        self.out_features = eyes_out
        self.sub_model = torch.nn.Sequential(
            # pyre-fixme[16]: Module `models` has no attribute `vgg16_bn`.
            *list(models.vgg16_bn().features.children())[1:]
        )
        self.new_first = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.classifier = torch.nn.Sequential(
            # pyre-fixme[16]: Module `models` has no attribute `vgg16_bn`.
            *list(models.vgg16_bn().classifier.children())[:-4]
        )
        # pyre-fixme[4]: Attribute must be annotated.
        # pyre-fixme[16]: Module `models` has no attribute `vgg16_bn`.
        self.avgpool = models.vgg16_bn().avgpool
        self.linear = nn.Linear(in_features=4096, out_features=256, bias=True)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        out = self.new_first(x)
        out = self.sub_model(out)
        out = self.avgpool(out)
        #         print(out.shape)
        out = out.view(-1, 7 * 7 * 512)
        #         print(out.shape)
        out = self.classifier(out)
        out = self.linear(out)
        return out


class EyesEncoderResnet(nn.Module):
    def __init__(self, eyes_out: int = 256) -> None:
        super(EyesEncoderResnet, self).__init__()
        self.out_features = eyes_out
        # pyre-fixme[4]: Attribute must be annotated.
        # pyre-fixme[16]: Module `models` has no attribute `resnet18`.
        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Linear(512, eyes_out)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, x):
        out = self.resnet18(x)
        return out


class GazeEstimationModel(nn.Module):
    def __init__(
        self,
        eyes_out: int = 256,
        kpts_out: int = 64,
        use_bias: bool = False,
        model_type: str = "vgg16",
        device: str = "cuda",
    ) -> None:
        super(GazeEstimationModel, self).__init__()
        assert model_type in ["vgg16", "resnet18"], "Wrong model type"

        self.model_type = model_type
        self.device = device
        if model_type == "vgg16":
            # pyre-fixme[4]: Attribute must be annotated.
            self.eyes_encoder = EyesEncoderVGG(eyes_out=eyes_out)

        elif model_type == "resnet18":
            self.eyes_encoder = EyesEncoderResnet(eyes_out=eyes_out)

        self.keypoints_encoder = KeypointsEncoder(
            num_inputs=68,
            num_harmonics=8,
            num_channels=128,
            num_layers=3,
            output_channels=kpts_out,
            output_size=1,
        )

        finale_fc = [
            nn.ReLU(inplace=False),
            #                  nn.Dropout(p=0.2, inplace=False),
            nn.Linear(eyes_out + kpts_out, 64, bias=use_bias),
            nn.ReLU(inplace=False),
            nn.Linear(64, 2, bias=use_bias),
        ]

        finale_fc_blink = [
            nn.ReLU(inplace=False),
            #                  nn.Dropout(p=0.2, inplace=False),
            nn.Linear(eyes_out, 64, bias=use_bias),
            nn.ReLU(inplace=False),
            nn.Linear(64, 1, bias=use_bias),
        ]
        self.final_model = nn.Sequential(*finale_fc)
        self.final_model_blink = nn.Sequential(*finale_fc_blink)
        self.prepared_eyes_encoder: Union[nn.Module, None] = None

    def get_eye_embeddings(
        self, left_eye: torch.Tensor, right_eye: torch.Tensor, layer_indices: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        def extract_embeddings(
            model: nn.Sequential, x: torch.Tensor
        ) -> List[torch.Tensor]:
            embeddings = []
            j = 0
            for i, layer in enumerate(model):
                x = layer(x)
                if i == layer_indices[j]:
                    embeddings.append(x)
                    j += 1
                    if j >= len(layer_indices):
                        break

            return embeddings

        if self.model_type == "vgg16":

            def prepare_features(model) -> nn.Sequential:
                result = [*([model.new_first] + list(model.sub_model))]
                return nn.Sequential(*result)

            self.prepared_eyes_encoder = prepare_features(self.eyes_encoder)
        elif self.model_type == "resnet18":

            def prepare_features(model: nn.Module) -> nn.Sequential:
                result: List[nn.Module] = [
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                    model.avgpool,
                ]
                return nn.Sequential(*result)

            self.prepared_eyes_encoder = prepare_features(self.eyes_encoder.resnet18)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        left_embeddings = extract_embeddings(
            # pyre-ignore
            self.prepared_eyes_encoder,
            left_eye.to(self.device),
        )
        right_embeddings = extract_embeddings(
            # pyre-ignore
            self.prepared_eyes_encoder,
            right_eye.to(self.device),
        )

        return left_embeddings, right_embeddings

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, left_eye, right_eye, kp2d, mode: str = "full"):
        encoded_l = self.eyes_encoder(left_eye)
        encoded_r = self.eyes_encoder(right_eye)
        if mode == "full":
            encoded_kpts = self.keypoints_encoder(kp2d)
            cat = torch.cat([encoded_l + encoded_r, encoded_kpts], dim=1)
            out = self.final_model(cat)
            cat_b = torch.cat([encoded_l + encoded_r], dim=1)
            out_blink = self.final_model_blink(cat_b)
            return out, out_blink, encoded_l, encoded_r, encoded_kpts
        elif mode == "gaze":
            encoded_kpts = self.keypoints_encoder(kp2d)
            cat = torch.cat([encoded_l + encoded_r, encoded_kpts], dim=1)
            out = self.final_model(cat)
            return out, encoded_l, encoded_r
        elif mode == "eyes_enc":
            return encoded_l, encoded_r
        else:
            raise ValueError("Wrong mode of model forward")
