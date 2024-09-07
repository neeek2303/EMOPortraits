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

from utils import args as args_utils
from utils.point_transforms import parse_3dmm_param



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
                 scale_cond_tr = 0.8
                 ):
        super(LMDBDataset, self).__init__()
        self.envs = []
        for i in range(128):
            self.envs.append(lmdb.open(f'{data_root}/{i}_lmdb', max_readers=1, readonly=True, 
                                       lock=False, readahead=False, meminit=False))
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

        self.epoch_len = epoch_len
        self.random_frames = random_frames
        self.align_source = align_source
        self.align_target = align_target
        self.align_scale = align_scale

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
        # random.seed(23)
        # Find split
        # if self.random_frames:
        #     index = random.randrange(0, self.splits[-1])

        if self.random_frames:
            a = len(self.splits) - index
            i = max(a, index)
            add_index = int(torch.randint(0, i, (1,))[0])
            index = index + add_index if a > index else index - add_index


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
        
        return data_dict

    def sample_data_dict(self, index, split, n, t):

        env = self.envs[split]

        for i in range(len(self.keys[split])):
            if len(self.keys[split][index]) >= t:
                break
            else:
                index = (index + i) % len(self.keys[split])

        if self.random_frames:
            keys = [self.keys[split][index][random.randrange(0, len(self.keys[split][index]))] for i in range(n)]

            i_start = random.randrange(0, len(self.keys[split][index]) + 1 - t)
            keys += [self.keys[split][index][i_start + j] for j in range(t)]

        else:
            keys = [self.keys[split][index][i] for i in range(n)] + [self.keys[split][index][-j] for j in reversed(range(1, t + 1))]
        
        data_dict = {
            'image': [],
            'mask':[],

            'size': [],
            'face_scale': [],
            'keypoints': [],
            'params_3dmm': {'R': [], 'offset': [], 'roi_box': [], 'size': []},
            'params_ffhq': {'theta': []},
            'crop_box': []}
    
        with env.begin(write=False) as txn:
            for key in keys:
                item = pickle.loads(txn.get(key))
                # print(type(item['image']), type(item['mask']), type(item['size']), type(item['face_scale']), type(item['keypoints_3d']), )
                image = Image.open(io.BytesIO(item['image'])).convert('RGB')
                mask = Image.open(io.BytesIO(item['mask']))


                data_dict['image'].append(image)
                data_dict['mask'].append(mask)

                data_dict['size'].append(item['size'])
                data_dict['face_scale'].append(item['face_scale'])
                data_dict['keypoints'].append(item['keypoints_3d'])

                R, offset, _, _ = parse_3dmm_param(item['3dmm']['param'])

                data_dict['params_3dmm']['R'].append(R)
                data_dict['params_3dmm']['offset'].append(offset)
                data_dict['params_3dmm']['roi_box'].append(item['3dmm']['bbox'])
                data_dict['params_3dmm']['size'].append(item['size'])

                data_dict['params_ffhq']['theta'].append(item['transform_ffhq']['theta'])

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

        if self.use_masked_aug:
            data_dict['masked_face'] = [np.where(np.expand_dims(np.asarray(m).copy(), -1)>250, np.asarray(img).copy(), 0).astype(np.uint8) for img, m in zip(data_dict['image'], data_dict['mask'])]
            # print(np.max(data_dict['masked_face'][0]), np.min(data_dict['masked_face'][0]), 'a',
            #       np.max(data_dict['image'][0]), np.min(data_dict['image'][0]), 'b',
            #       np.max(data_dict['mask'][0]), np.min(data_dict['mask'][0]), 'c')

        # Augment with local warpings
        if self.output_aug_warp:
            if self.use_masked_aug:
                warp_aug = self.augment_via_warp(data_dict['masked_face'], self.aug_warp_size)
            else:
                warp_aug = self.augment_via_warp(data_dict['image'], self.aug_warp_size)

            warp_aug = torch.stack([self.to_tensor(w) for w in warp_aug], dim=0)

        imgs = torch.stack([self.to_tensor(img) for img in data_dict['image']])
        masks = torch.stack([self.to_tensor(m) for m in data_dict['mask']])
        # print(imgs.shape)
        keypoints = torch.FloatTensor(data_dict['keypoints'])

        R = torch.FloatTensor(data_dict['params_3dmm']['R'])
        offset = torch.FloatTensor(data_dict['params_3dmm']['offset'])
        roi_box = torch.FloatTensor(data_dict['params_3dmm']['roi_box'])[:, None]
        size = torch.FloatTensor(data_dict['params_3dmm']['size'])[:, None, None]
        theta = torch.FloatTensor(data_dict['params_ffhq']['theta'])
        crop_box = torch.FloatTensor(data_dict['crop_box'])[:, None]



        if self.align_source or self.align_target:
            # Align input images using theta
            eye_vector = torch.zeros(theta.shape[0], 1, 3)
            eye_vector[:, :, 2] = 1

            theta_ = torch.cat([theta, eye_vector], dim=1).float()

            # Perform 2x zoom-in compared to default theta
            scale = torch.zeros_like(theta_)
            scale[:, [0, 1], [0, 1]] = self.align_scale
            scale[:, 2, 2] = 1

            theta_ = torch.bmm(theta_, scale)[:, :2]

            align_warp = self.identity_grid.repeat_interleave(theta_.shape[0], dim=0)
            align_warp = align_warp.bmm(theta_.transpose(1, 2)).view(theta_.shape[0], self.image_size, self.image_size, 2)

            if self.align_source:
                imgs[:n] = F.grid_sample(imgs[:n], align_warp[:n])
                masks[:n] = F.grid_sample(masks[:n], align_warp[:n])
                if self.output_aug_warp:
                    warp_aug[:n] = F.grid_sample(warp_aug[:n], align_warp[:n])

            if self.align_target:
                imgs[-t:] = F.grid_sample(imgs[-t:], align_warp[-t:])
                masks[-t:] = F.grid_sample(masks[-t:], align_warp[-t:])
                if self.output_aug_warp:
                    warp_aug[-t:] = F.grid_sample(warp_aug[-t:], align_warp[-t:])


        output_data_dict = {
            'source_img': imgs[:n],
            'source_mask': masks[:n],
            'source_keypoints': keypoints[:n],
            # 'source_params_3dmm': {
            #     'R': R[:n],
            #     'offset': offset[:n],
            #     'roi_box': roi_box[:n],
            #     'size': size[:n],
            #     'crop_box': crop_box[:n]},

            # 'source_params_ffhq': {
            #     'theta': theta[:n],
            #     'crop_box': crop_box[:n]},

            'target_img': imgs[-t:],
            'target_mask': masks[-t:],
            'target_keypoints': keypoints[-t:],

            # 'target_params_3dmm': {
            #     'R': R[-t:],
            #     'offset': offset[-t:],
            #     'roi_box': roi_box[-t:],
            #     'size': size[-t:],
            #     'crop_box': crop_box[-t:]},

            # 'target_params_ffhq': {
            #     'theta': theta[-t:],
            #     'crop_box': crop_box[-t:]}
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
        parser.add_argument('--data_root', default='/fsx/VC2_HD_f', type=str)
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
        self.data_root = args.data_root
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

        for i in range(128):


            keys_i = pickle.load(open(f'{self.data_root}/{i}_lmdb/keys_best.pkl', 'rb'))

            for phase, keys_phase in keys_i.items():
                keys_phase_list = []

                for keys_video in keys_phase:
                    keys_video_list = []

                    for key_start, num_keys in keys_video:
                        parts = key_start.split('/')
                        frame_start = int(parts[-1])

                        for i in range(num_keys):
                            frame = '%06d' % (frame_start + i)
                            parts[-1] = frame
                            keys_video_list.append('/'.join(parts).encode())

                    keys_phase_list.append(keys_video_list)

                self.keys[phase].append(keys_phase_list)
                self.splits[phase].append(len(keys_phase_list))
        
        for phase in ['test', 'train']:
            self.splits[phase] = np.cumsum(np.asarray(self.splits[phase]))

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
                                    scale_cond_tr = self.args.scale_cond_tr)

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