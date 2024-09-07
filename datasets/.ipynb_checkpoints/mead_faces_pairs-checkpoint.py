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
import torch.nn.functional as F
from random import choice

from utils import args as args_utils
from utils.point_transforms import parse_3dmm_param



list_of_lmdbs = [

            'M005/front', 'M005/left_30', 'M005/right_30',
            'M007/front', 'M007/left_30', 'M007/right_30',
            'M009/front', 'M009/left_30', 'M009/right_30',
            'M011/front', 'M011/left_30', 'M011/right_30',
            'M012/front', 'M012/left_30', 'M012/right_30',
            'M013/front', 'M013/left_30', 'M013/right_30',
            'M022/front', 'M022/left_30', 'M022/right_30',
            'M023/front', 'M023/left_30', 'M023/right_30',
            'M024/front', 'M024/left_30', 'M024/right_30',
            'M025/front', 'M025/left_30', 'M025/right_30',
            'M027/front', 'M009/left_30', 'M009/right_30',
            'M028/front', 'M011/left_30', 'M011/right_30',
            'M029/front', 'M012/left_30', 'M012/right_30',
            'M030/front', 'M013/left_30', 'M013/right_30',
            'M031/front', 'M022/left_30', 'M022/right_30',
            'M032/front', 'M023/left_30', 'M023/right_30',
            'M033/front', 'M024/left_30', 'M024/right_30',
            'M034/front', 'M025/left_30', 'M025/right_30',

            # 'M035/front', 'M035/left_30', 'M035/right_30',
            'M037/front', 'M037/left_30', 'M037/right_30',
            'M039/front', 'M039/left_30', 'M039/right_30',
            'M040/front', 'M040/left_30', 'M040/right_30',
            'M041/front', 'M041/left_30', 'M041/right_30',
            'M042/front', 'M042/left_30', 'M042/right_30',
            'M043/front', 'M043/left_30', 'M043/right_30',




            'W009/front', 'W009/left_30', 'W009/right_30',
            'W011/front', 'W011/left_30', 'W011/right_30',
            'W014/front', 'W014/left_30', 'W014/right_30',
            'W015/front', 'W015/left_30', 'W015/right_30',
            'W016/front', 'W016/left_30', 'W016/right_30',
            'W018/front', 'W018/left_30', 'W018/right_30',
            'W019/front', 'W019/left_30', 'W019/right_30',
            'W021/front', 'W021/left_30', 'W021/right_30',
            'W023/front', 'W023/left_30', 'W023/right_30',
            'W025/front', 'W025/left_30', 'W025/right_30',
            'W026/front', 'W026/left_30', 'W026/right_30',
            'W029/front', 'W029/left_30', 'W029/right_30',
            'W033/front', 'W033/left_30', 'W033/right_30',
            'W035/front', 'W035/left_30', 'W035/right_30',
            'W036/front', 'W036/left_30', 'W036/right_30',
            'W037/front', 'W037/left_30', 'W037/right_30',
            'W038/front', 'W038/left_30', 'W038/right_30',
            'W039/front', 'W039/left_30', 'W039/right_30',




            'M005/left_60', 'M005/right_60',
            'M012/left_60', 'M012/right_60',
            'M022/left_60', 'M022/right_60',
            'M037/left_60', 'M037/right_60',
            
            'W009/left_60', 'W009/right_60',
            'W015/left_60', 'W015/right_60',
            'W018/left_60', 'W018/right_60',
            'W016/left_60', 'W016/right_60',
            'W019/left_60', 'W019/right_60',
            'W021/left_60', 'W021/right_60',
            'W023/left_60', 'W023/right_60',
            'W025/left_60', 'W025/right_60',
            'M003/front', 'M003/left_30', 'M003/right_30',
            
        ]






list_of_lmdbs_only_test = [
            # 'Rodrigo/RANDOM',
            'M003/right_30',
        ]




class LMDBDataset(data.Dataset):
    def __init__(self, 
                 data_root, 
                 num_source_frames, 
                 num_target_frames,
                 image_size,
                 image_additional_size,
                 keys,
                 splits,
                 augment_geometric = False, 
                 augment_color = False, 
                 output_aug_warp = False,
                 output_aug_warp_out = True,
                 use_masked_aug = False,
                 aug_warp_size = -1,
                 epoch_len = -1,
                 random_frames = False,
                 align_source = False,
                 align_target = False,
                 align_scale = 1.33,
                 rot_aug_angle = 0,
                 rand_crop_prob = 0,
                 rand_shift_prob = 0,
                 rand_crop_scale = 0.9,
                 warp_aug_color_coef = 1.0,
                 aug_color_coef = 1.0,
                 gray_source_prob = 0.0,
                 shift_limit = 0.0,
                 scale_cond_tr = 0.8,
                 excl_window_len=200,
                 splits_names=False,
                 lmdbs_list = [],
                 splits_env_index = False
                 ):
        super(LMDBDataset, self).__init__()
        self.envs = []
        self.envs_names = []

        self.splits_names = splits_names
        for i in lmdbs_list:
            self.envs.append(lmdb.open(f'{data_root}/lmdbs/{i}', max_readers=1, readonly=True, 
                                       lock=False, readahead=False, meminit=False))
            self.envs_names.append(i)

        self.splits_env_index = splits_env_index  
        self.excl_window_len = excl_window_len
        self.keys = keys
        self.scale_cond_tr = scale_cond_tr
        self.splits = splits
        self.shift_limit = shift_limit
        self.aug_color_coef = aug_color_coef
        self.warp_aug_color_coef = warp_aug_color_coef
        self.num_source_frames = num_source_frames
        self.num_target_frames = num_target_frames
        self.gray_source_prob = gray_source_prob
        self.image_size = image_size
        self.rot_aug_angle = rot_aug_angle
        self.rand_crop_prob = rand_crop_prob
        self.rand_crop_scale = rand_crop_scale
        self.rand_shift_prob = rand_shift_prob
        self.image_additional_size = image_additional_size
        # print(self.image_additional_size, self.image_size)
        self.augment_geometric = augment_geometric
        self.augment_color = augment_color
        self.output_aug_warp = output_aug_warp
        self.output_aug_warp_out = output_aug_warp_out
        self.use_masked_aug = use_masked_aug
        self.to_image = transforms.ToPILImage()

        self.epoch_len = epoch_len
        self.random_frames = random_frames
        self.align_source = align_source
        self.align_target = align_target
        self.align_scale = align_scale
        self.prev_index = 1

        if self.align_source:
            grid = torch.linspace(-1, 1, self.image_size)
            v, u = torch.meshgrid(grid, grid)
            self.identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)

        # Transformsf
        if self.augment_color:
            self.aug = A.Compose(
                [A.ColorJitter(hue=0.03*self.aug_color_coef,
                               brightness=0.06 * max(1, self.aug_color_coef/2),
                               contrast=0.03*self.aug_color_coef,
                               saturation=0.03*self.aug_color_coef,
                               p=0.8),
                A.ToGray(p=self.gray_source_prob)
                 # A.Rotate(limit=5)
                 ],
                # additional_targets={f'image{k}': 'image' for k in range(1, num_source_frames + num_target_frames)})
                additional_targets={f'image{k}': 'image' for k in range(1, num_source_frames + num_target_frames)})
            




            self.rot_aug = A.Compose(
                [
                 A.Rotate(limit=self.rot_aug_angle, value=0)
                 ],additional_targets={'mask': 'image', 'mask1':'image1'})
            
            self.rand_crop = A.Compose(
                [
                 A.ShiftScaleRotate(shift_limit=self.shift_limit, scale_limit=0.0, rotate_limit=0, interpolation=1, border_mode=0, value=0, 
                                    mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=self.rand_shift_prob),

                 A.RandomResizedCrop(height=512, width=512, scale=(self.rand_crop_scale, 1.0), ratio=(1, 1), p=self.rand_crop_prob)
                 ], additional_targets={'mask': 'image', 'mask1':'image1'})



            self.flip = A.ReplayCompose(
                [A.HorizontalFlip(p=0.5)
                 ],
                 keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
                # additional_targets={f'image{k}': 'image' for k in range(1, num_source_frames + num_target_frames)})
                additional_targets={'image1': 'image', 'mask': 'image', 'mask1':'image', 'keypoints':'keypoints', 'keypoints1':'keypoints'})
        self.to_tensor = transforms.ToTensor()

        if self.output_aug_warp:
            self.aug_warp_size = aug_warp_size

            # Greate a uniform meshgrid, which is used for warping calculation from deltas
            tick = torch.linspace(0, 1, self.aug_warp_size)
            v, u = torch.meshgrid(tick, tick)
            grid = torch.stack([u, v, torch.zeros(self.aug_warp_size, self.aug_warp_size)], dim=2)
            
            self.grid = (grid * 255).numpy().astype('uint8') # aug_warp_size x aug_warp_size x 3

    @staticmethod
    def to_tensor_keypoints(keypoints, size):
        keypoints = torch.from_numpy(keypoints).float()
        keypoints /= size
        keypoints[..., :2] -= 0.5
        keypoints *= 2

        return keypoints
    
    @staticmethod
    def to_image_keypoints(keypoints, size=512):
        keypoints /= 2
        keypoints[..., :2] += 0.5
        keypoints *= size
        
        return torch.tensor(keypoints).numpy()
    
    @staticmethod
    def from_image_keypoints(keypoints, size=512):
        keypoints = torch.tensor(keypoints).float()
        keypoints /= size
        keypoints[..., :2] -= 0.5
        keypoints *= 2

        return keypoints.float()

    def __getitem__(self, index):
        n = self.num_source_frames
        t = self.num_target_frames

        
        # Find split
        if self.random_frames:
            add_seed = random.randrange(0, self.splits[-1])
            random.seed(index*self.prev_index+add_seed)
            index = random.randrange(0, self.splits[-1])

        # if self.random_frames:
        #     a = len(self.splits) - index
        #     i = max(a, index)
        #     add_index = int(torch.randint(0, i, (1,))[0])
        #     index = index + add_index if a > index else index - add_index

        split = np.where(self.splits > index)[0][0]
        if split > 0:
            index = index - self.splits[split - 1]
        
        g=0
        ng=0
        while g == 0:
            try:
                data_dict  = self.sample_data_dict(index, split, n, t)
                g=1
                ng=0
            except Exception as e:
                random.seed(index+ng)
                index = random.randrange(0, self.splits[-1])
                split = np.where(self.splits > index)[0][0]
                if split > 0:
                    index = index - self.splits[split - 1]
                ng+=1
                print(index, ng)
                print(e)

        self.prev_index = index

        return data_dict

    
    def sample_data_dict(self, index, split, n, t):

    
        for i in range(len(self.keys[split])):
            if len(self.keys[split][index]) >= t:
                break
            else:
                index = (index + i) % len(self.keys[split])
        # print(self.keys[split][index].decode('ascii').split('/'))

              
        env = self.envs[self.splits_env_index[self.splits_names[split]]]

        if not self.random_frames:
            random.seed(23)
        LL = len(self.keys[split])
        exclud_set = set([ min(LL-1, max(index-self.excl_window_len//2, 0)) for i in range(self.excl_window_len)])
        target_indx = choice([i for i in range(0, LL) if i not in exclud_set])
        keys = [self.keys[split][index],  self.keys[split][target_indx]] 


        data_dict = {
            'image': [],
            'mask':[],
            'size': [],
            'face_scale': [],
            'keypoints': [],
            'crop_box': []}
        # print('nnnnnnnnnnnnnn')
        # print(self.envs_names)
        with env.begin(write=False) as txn:
            for key in keys:

                try:
                    item = pickle.loads(txn.get(key))
                except Exception as e:
                    print(self.envs_names[self.splits_env_index[self.splits_names[split]]], self.splits_env_index[self.splits_names[split]], self.splits_names[split], split,  key)
                
    
                item = pickle.loads(txn.get(key))
                image = self.to_image(np.uint8(np.moveaxis(item['image'], 0, 2)*255))
                mask = self.to_image(np.uint8(np.moveaxis(item['mask'], 0, 2)*255))


                data_dict['image'].append(image)
                data_dict['mask'].append(mask)

                data_dict['size'].append(item['size'])
                data_dict['face_scale'].append(item['scale'])
                data_dict['keypoints'].append(item['keypoints'])


        # Geometric augmentations and resize
        data_dict = self.preprocess_data(data_dict)
        data_dict['image'] = [np.asarray(img).copy() for img in data_dict['image']]
        data_dict['mask'] = [np.asarray(m).copy() for m in data_dict['mask']]


        # Augment color
        if self.augment_color:
            imgs_dict = {(f'image{k}' if k > 0 else 'image'): img for k, img in enumerate(data_dict['image'])}
            data_dict['image'] = list(self.aug(**imgs_dict).values())
            if self.rot_aug_angle > 0:
                imgs_mask_dict = {'image':data_dict['image'][0], 'image1':data_dict['image'][1], 'mask':data_dict['mask'][0], 'mask1':data_dict['mask'][1]}
                flipped = self.rot_aug(**imgs_mask_dict)
                data_dict['image'] = [flipped['image'], flipped['image1']]
                data_dict['mask'] = [flipped['mask'], flipped['mask1']]

            scale_cond = False
            for sc in data_dict['face_scale']:
                scale_cond+=sc>=self.scale_cond_tr


            if (self.rand_crop_prob>0.0 or self.rand_shift_prob>0.0) and scale_cond:
                imgs_mask_dict = {'image':data_dict['image'][0], 'image1':data_dict['image'][1], 'mask':data_dict['mask'][0], 'mask1':data_dict['mask'][1]}
                flipped = self.rand_crop(**imgs_mask_dict)
                data_dict['image'] = [flipped['image'], flipped['image1']]
                data_dict['mask'] = [flipped['mask'], flipped['mask1']]


            imgs_mask_dict = {'image':data_dict['image'][0], 'image1':data_dict['image'][1], 'mask':data_dict['mask'][0], 'mask1':data_dict['mask'][1], 'keypoints':self.to_image_keypoints(data_dict['keypoints'][0]), 'keypoints1':self.to_image_keypoints(data_dict['keypoints'][1])}
            # flipped = self.flip(**imgs_mask_dict)
            flipped = self.flip(image = imgs_mask_dict['image'], image1 = imgs_mask_dict['image1'], mask = imgs_mask_dict['mask'], mask1 = imgs_mask_dict['mask1'], keypoints = imgs_mask_dict['keypoints'], keypoints1 = imgs_mask_dict['keypoints1'])
            data_dict['image'] = [flipped['image'], flipped['image1']]
            data_dict['mask'] = [flipped['mask'], flipped['mask1']]
            data_dict['keypoints'] = torch.stack([self.from_image_keypoints(flipped['keypoints']), self.from_image_keypoints(flipped['keypoints1'])], dim=0)


        # Augment with local warpings
        if self.output_aug_warp:
            if self.use_masked_aug:
                warp_aug = self.augment_via_warp(data_dict['masked_face'], self.aug_warp_size)
            else:
                warp_aug = self.augment_via_warp(data_dict['image'], self.aug_warp_size)

            warp_aug = torch.stack([self.to_tensor(w) for w in warp_aug], dim=0)

        imgs = torch.stack([self.to_tensor(img) for img in data_dict['image']])
        masks = torch.stack([self.to_tensor(m) for m in data_dict['mask']])

        keypoints = torch.FloatTensor(data_dict['keypoints'])


        output_data_dict = {
            'source_img': imgs[:n],
            'source_mask': masks[:n],
            'source_keypoints': keypoints[:n],


            'target_img': imgs[-t:],
            'target_mask': masks[-t:],
            'target_keypoints': keypoints[-t:],

        }

        if self.output_aug_warp and self.output_aug_warp_out:

            # trans = transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(), ]), p = 0.3)
            self.aug_d_gray =  transforms.RandomGrayscale(p=0.05)
            # self.aug_d_gray = transforms.Grayscale(num_output_channels=3)
            self.aug_d = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4*self.warp_aug_color_coef,
            saturation=0.4*self.warp_aug_color_coef,
            hue=0.4*self.warp_aug_color_coef)

            # self.aug_dd = transforms.ColorJitter(
            # brightness=0.2,
            # contrast=0.1,
            # saturation=0.1,
            # hue=0.1)

            output_data_dict['source_warp_aug'] = self.aug_d_gray(self.aug_d(warp_aug[:n]))
            output_data_dict['target_warp_aug'] = self.aug_d_gray(self.aug_d(warp_aug[-t:]))

        # if self.output_aug_warp and self.output_aug_warp_out:
        #     output_data_dict['source_warp_aug'] = warp_aug[:n]
        #     self.aug_d = transforms.ColorJitter(
        #     brightness=0.2,
        #     contrast=0.2,
        #     saturation=0.2,
        #     hue=0.2)
        #     output_data_dict['target_warp_aug'] = self.aug_d(warp_aug[-t:])

        # else:
        #     if self.use_masked_aug:
        #         output_data_dict['source_warp_aug'] = data_dict['masked_face'][:n]
        #         output_data_dict['target_warp_aug'] = data_dict['masked_face'][-t:]
        #     else:
        #         output_data_dict['source_warp_aug'] = imgs[:n]
        #         output_data_dict['target_warp_aug'] = imgs[-t:]

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
                pass # use params of the previous frame

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
            if self.image_additional_size != self.image_size:
                image = image.resize((self.image_additional_size, self.image_additional_size), Image.BICUBIC)
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

    @staticmethod
    def augment_via_warp(images, image_size):
        # Implementation is based on DeepFaceLab repo
        # https://github.com/iperov/DeepFaceLab
        # 
        # Performs an elastic-like transform for a uniform grid accross the image
        image_aug = []

        for image in images:
            cell_count = 8 + 1
            cell_size = image_size // (cell_count - 1)

            grid_points = np.linspace(0, image_size, cell_count)
            mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
            mapy = mapx.T

            mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] + np.random.normal(size=(cell_count-2, cell_count-2)) * cell_size * 0.1
            mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] + np.random.normal(size=(cell_count-2, cell_count-2)) * cell_size * 0.1

            half_cell_size = cell_size // 2

            mapx = cv2.resize(mapx, (image_size + cell_size,) * 2)[half_cell_size:-half_cell_size, half_cell_size:-half_cell_size].astype(np.float32)
            mapy = cv2.resize(mapy, (image_size + cell_size,) * 2)[half_cell_size:-half_cell_size, half_cell_size:-half_cell_size].astype(np.float32)

            image_aug += [cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)]

        return image_aug
                
    def __len__(self):
        if self.epoch_len == -1:
            return self.splits[-1]
        else:
            return min(self.epoch_len, self.splits[-1])


class DataModule(object):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("dataset")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--test_batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--data_root', default='/fsx/behavioural_computing_data/face_generation_data/MEAD', type=str)
        parser.add_argument('--num_source_frames', default=1, type=int)
        parser.add_argument('--num_target_frames', default=1, type=int)
        parser.add_argument('--rot_aug_angle', default=0.0, type=float)
        parser.add_argument('--warp_aug_color_coef', default=1.0, type=float)
        parser.add_argument('--aug_color_coef', default=1.0, type=float)
        parser.add_argument('--gray_source_prob', default=0.0, type=float)
        parser.add_argument('--shift_limit', default=0.1, type=float)
        parser.add_argument('--scale_cond_tr', default=0.8, type=float)
        
        
        parser.add_argument('--rand_crop_prob', default=0.0, type=float)
        parser.add_argument('--rand_shift_prob', default=0.0, type=float)
        parser.add_argument('--rand_crop_scale', default=0.9, type=float)
        parser.add_argument('--image_size', default=256, type=int)
        parser.add_argument('--image_additional_size', default=None, type=int)
        parser.add_argument('--image_additional_size_d', default=None, type=int)
        parser.add_argument('--aug_warp_size', default=256, type=int)
        parser.add_argument('--augment_geometric_train', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--augment_color_train', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--output_aug_warp', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--output_aug_warp_out', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_masked_aug', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_hq', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--use_diverse', default='True', type=args_utils.str2bool, choices=[True, False])


        # These parameters can be used for debug
        parser.add_argument('--train_epoch_len', default=-1, type=int)
        parser.add_argument('--test_epoch_len', default=-1, type=int)

        return parser_out

    def __init__(self, args):
        super(DataModule, self).__init__()
        self.args = args
        self.ddp = args.num_gpus > 1
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.data_root = '/fsx/behavioural_computing_data/face_generation_data/MEAD'
        self.num_source_frames = args.num_source_frames
        self.num_target_frames = args.num_target_frames
        self.image_size = args.image_size
        self.rot_aug_angle = args.rot_aug_angle
        self.rand_crop_prob = args.rand_crop_prob
        self.rand_crop_scale = args.rand_crop_scale
        self.image_additional_size = args.image_additional_size_d
        if self.image_additional_size is None:
            self.image_additional_size = self.image_size

        self.augment_geometric_train = args.augment_geometric_train
        self.augment_color_train = args.augment_color_train
        self.output_aug_warp = args.output_aug_warp
        self.output_aug_warp_out = args.output_aug_warp_out
        self.use_masked_aug = args.use_masked_aug
        self.aug_warp_size = args.aug_warp_size
        self.train_epoch_len = args.train_epoch_len
        self.test_epoch_len = args.test_epoch_len

        self.keys = {'test': [], 'train': []}
        self.splits = {'test': [], 'train': []}
        self.splits_names = {'test': [], 'train': []}
        self.splits_env_index = {'test': {}, 'train': {}}
        self.video_names = []
        env_indx_train = 0
        env_indx_test = 0
        for L_n in list_of_lmdbs:
            keys_i = pickle.load(open(f'{self.data_root}/lmdbs/{L_n}_keys_dict.pkl', 'rb'))
            for phase, keys_phase in keys_i.items():
                if (not (L_n in list_of_lmdbs_only_test) and phase=='train'):
                    for vid_name in keys_i[phase].keys():
                 
                        keys_vid_list = []
                        if len(keys_i[phase][vid_name])>0:
                            for key in keys_i['train'][vid_name]: # for every
                                keys_vid_list.append(key.encode())
                            self.video_names.append(vid_name)
                            self.keys[phase].append(keys_vid_list)
                            self.splits[phase].append(len(keys_vid_list))
                            self.splits_names[phase].append(L_n)
                            self.splits_env_index[phase].update({L_n:env_indx_train})
                    env_indx_train+=1

                elif (L_n in list_of_lmdbs_only_test and phase=='test'):
                    for vid_name in keys_i[phase].keys():
                      
                        # if '_C' in vid_name:
                        keys_vid_list = []
                        if len(keys_i[phase][vid_name])>0:
                            for key in keys_i['train'][vid_name]: # for every
                                keys_vid_list.append(key.encode())
                            self.video_names.append(vid_name)
                            self.keys[phase].append(keys_vid_list)
                            self.splits[phase].append(len(keys_vid_list))
                            self.splits_names[phase].append(L_n)
                            self.splits_env_index[phase].update({L_n:env_indx_test})
                    env_indx_test+=1


        # for L_n in list_of_lmdbs + list_of_lmdbs_only_test:
        #     keys_i = pickle.load(open(f'{self.data_root}/lmdbs/{L_n}_keys_dict.pkl', 'rb'))
        #     for phase, keys_phase in keys_i.items():
        #         for vid_name in keys_i[phase].keys():
        #             keys_vid_list = []
        #             if len(keys_i[phase][vid_name])>0:
        #                 phase = 'test' if L_n in list_of_lmdbs_only_test else 'train'
        #                 for key in keys_i['train'][vid_name] + keys_i['test'][vid_name]: # for every
        #                     keys_vid_list.append(key.encode())

        #                 self.keys[phase].append(keys_vid_list)
        #                 self.splits[phase].append(len(keys_vid_list))
        #                 self.splits_names[phase].append(L_n)


        for phase in ['test', 'train']:
            self.splits[phase] = np.cumsum(np.asarray(self.splits[phase]))

        # print(self.splits_names['train'])
        # print(self.splits_names['test'])
        # print(self.video_names)
        
    def train_dataloader(self):
        train_dataset = LMDBDataset(self.data_root,
                                    self.num_source_frames, 
                                    self.num_target_frames,
                                    self.image_size,
                                    self.image_additional_size,
                                    self.keys['train'],
                                    self.splits['train'],
                                    self.augment_geometric_train,
                                    self.augment_color_train,
                                    self.output_aug_warp,
                                    self.output_aug_warp_out,
                                    self.use_masked_aug,
                                    self.aug_warp_size,
                                    self.train_epoch_len,
                                    random_frames=True,
                                    rot_aug_angle = self.rot_aug_angle,
                                    rand_crop_prob = self.rand_crop_prob,
                                    rand_crop_scale = self.rand_crop_scale,
                                    warp_aug_color_coef = self.args.warp_aug_color_coef,
                                    aug_color_coef = self.args.aug_color_coef,
                                    gray_source_prob = self.args.gray_source_prob,
                                    shift_limit = self.args.shift_limit,
                                    rand_shift_prob = self.args.rand_shift_prob,
                                    scale_cond_tr = self.args.scale_cond_tr,
                                    splits_names = self.splits_names['train'],
                                    lmdbs_list=list_of_lmdbs,
                                    splits_env_index =self.splits_env_index['train'])

        shuffle = True
        sampler = None
        if self.ddp:
            shuffle = False
            sampler = data.distributed.DistributedSampler(train_dataset)

        return data.DataLoader(train_dataset, 
        					   batch_size=self.batch_size, 
        					   num_workers=self.num_workers, 
        					   pin_memory=True,
        					   shuffle=shuffle,
                               sampler=sampler)

    def test_dataloader(self):
        test_dataset = LMDBDataset(self.data_root,
                                   self.num_source_frames, 
                                   min(self.num_target_frames, 2),
                                   self.image_size,
                                   self.image_additional_size,
                                   self.keys['test'],
                                   self.splits['test'],
                                   epoch_len=self.test_epoch_len,
                                   lmdbs_list=list_of_lmdbs_only_test,
                                   splits_env_index =self.splits_env_index['test']
                                   )

        sampler = None
        if self.ddp:
            sampler = data.distributed.DistributedSampler(test_dataset, shuffle=True, seed=5) #4 or 5

        return data.DataLoader(test_dataset, 
        					   batch_size=self.test_batch_size, 
        					   num_workers=self.num_workers, 
        					   pin_memory=True,
                               sampler=sampler,
                               drop_last=True)