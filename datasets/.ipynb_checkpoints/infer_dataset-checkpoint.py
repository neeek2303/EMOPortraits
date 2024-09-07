import lmdb
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import albumentations as A
from argparse import ArgumentParser
import io
from PIL import Image, ImageOps
import random
import cv2
import pickle

from utils import args as args_utils
from utils.point_transforms import parse_3dmm_param


class InferDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 num_source_frames,
                 num_target_frames,
                 image_size,
                 augment_geometric=False,
                 augment_color=False,
                 output_aug_warp=False,
                 use_masked_aug=False,
                 epoch_len=-1,
                 random_frames=False,
                 ):
        super(InferDataset, self).__init__()

        self.num_source_frames = num_source_frames
        self.num_target_frames = num_target_frames
        self.image_size = image_size

        self.augment_geometric = augment_geometric
        self.augment_color = augment_color
        self.output_aug_warp = output_aug_warp
        self.use_masked_aug = use_masked_aug

        self.epoch_len = epoch_len
        self.random_frames = random_frames

        self.to_tensor = transforms.ToTensor()


    @staticmethod
    def to_tensor_keypoints(keypoints, size):
        keypoints = torch.from_numpy(keypoints).float()
        keypoints /= size
        keypoints[..., :2] -= 0.5
        keypoints *= 2

        return keypoints

    def __getitem__(self, index):
        n = self.num_source_frames
        t = self.num_target_frames

        data_dict = {
            'image': [],
            'mask': [],
            'keypoints': [],

            'crop_box': []}


        for key in keys:
            image = Image.open(io.BytesIO(item['image'])).convert('RGB')
            mask = Image.open(io.BytesIO(item['mask']))

            data_dict['image'].append(image)
            data_dict['mask'].append(mask)

            data_dict['size'].append(item['size'])
            data_dict['face_scale'].append(item['face_scale'])
            data_dict['keypoints'].append(item['keypoints_3d'])

        # Geometric augmentations and resize
        data_dict = self.preprocess_data(data_dict)
        data_dict['image'] = [np.asarray(img).copy() for img in data_dict['image']]
        data_dict['mask'] = [np.asarray(m).copy() for m in data_dict['mask']]



        imgs = torch.stack([self.to_tensor(img) for img in data_dict['image']])
        masks = torch.stack([self.to_tensor(m) for m in data_dict['mask']])

        keypoints = torch.FloatTensor(data_dict['keypoints'])

        crop_box = torch.FloatTensor(data_dict['crop_box'])[:, None]

        output_data_dict = {
            'source_img': imgs_s,
            'source_mask': masks_s,
            'source_keypoints': keypoints_s,

            'target_img': imgs_t,
            'target_mask': masks_t,
            'target_keypoints': keypoints_t,
        }


        return output_data_dict

    def preprocess_data(self, data_dict):
        MIN_SCALE = 0.67
        n = self.num_source_frames
        t = self.num_target_frames

        for i in range(len(data_dict['image'])):
            image = data_dict['image'][i]
            size = data_dict['size'][i]
            mask = data_dict['mask'][i]
            face_scale = data_dict['face_scale'][i]
            keypoints = data_dict['keypoints'][i]

            if i < n + 1:
                if self.augment_geometric and face_scale >= MIN_SCALE:
                    # Random sized crop
                    min_scale = MIN_SCALE / face_scale
                    seed = random.random()
                    scale = seed * (1 - min_scale) + min_scale
                    translate_x = random.random() * (1 - scale)
                    translate_y = random.random() * (1 - scale)

                else:
                    translate_x = 0
                    translate_y = 0
                    scale = 1

            else:
                pass  # use params of the previous frame

            crop_box = (size * translate_x,
                        size * translate_y,
                        size * (translate_x + scale),
                        size * (translate_y + scale))

            size_box = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

            keypoints[..., 0] = (keypoints[..., 0] - crop_box[0]) / size_box[0] - 0.5
            keypoints[..., 1] = (keypoints[..., 1] - crop_box[1]) / size_box[1] - 0.5
            keypoints[..., 2] = keypoints[..., 2] / (size_box[0] + size_box[1]) * 2
            keypoints *= 2

            data_dict['keypoints'][i] = keypoints

            image = image.crop(crop_box)
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            data_dict['image'][i] = image

            mask = mask.crop(crop_box)
            mask = mask.resize((self.image_size, self.image_size), Image.BICUBIC)
            data_dict['mask'][i] = mask

            # Normalize crop_box to work with coords in [-1, 1]
            crop_box = ((translate_x - 0.5) * 2,
                        (translate_y - 0.5) * 2,
                        (translate_x + scale - 0.5) * 2,
                        (translate_y + scale - 0.5) * 2)

            data_dict['crop_box'].append(crop_box)

        return data_dict


    def __len__(self):
        return self.pathes[-1]
