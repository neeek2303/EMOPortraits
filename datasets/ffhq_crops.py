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
from glob import glob
from utils import args as args_utils
from utils.point_transforms import parse_3dmm_param



class FFHQ_Crops_Dataset(data.Dataset):
    def __init__(self,
                 filtered_indexes,
                 main_folder,
                 images_root,
                 masks_root,
                 kp_root,
                 image_size,
                 augment_geometric = False,
                 augment_color = False,
                 output_aug_warp = False,
                 use_masked_aug = False,
                 aug_warp_size = -1,
                 epoch_len = -1,
                 random_frames = False,
                 augment_rotate=True,
                 augment_flip=False,
                 ):
        super(FFHQ_Crops_Dataset, self).__init__()



        self.main_folder = main_folder
        self.img_pathes = images_root
        self.masks_pathes = masks_root
        self.kp_pathes = kp_root

        self.img_pathes_sorted = sorted(glob(self.main_folder + f'/{images_root}' + '/*'))
        self.masks_pathes_sorted = sorted(glob(self.main_folder + f'/{masks_root}'  + '/*'))
        self.kp_pathes_sorted = sorted(glob(self.main_folder + f'/{kp_root}'  + '/*'))

        self.filtered_indexes_pathes = np.load(self.main_folder + f'/{filtered_indexes}')

        # assert len(self.img_pathes) == len(self.masks_pathes)
        # assert len(self.img_pathes) == len(self.kp_pathes)
        #
        # assert len(self.img_pathes) == len(self.masks_pathes), f'different number of files in ffhq folders {len(self.img_pathes)} != {len(self.masks_pathes)}'
        # random_indexes = [random.randint(0, len(self.img_pathes)-1) for i in range(10)]
        # assert [self.img_pathes[i].split('/')[-1] for i in random_indexes] == [self.masks_pathes[i].split('/')[-1] for i in random_indexes], 'img and mask are not aligned '

        self.image_size = image_size

        self.augment_geometric = augment_geometric
        self.augment_color = augment_color
        self.augment_rotate = augment_rotate
        self.augment_flip = augment_flip
        self.output_aug_warp = output_aug_warp
        self.use_masked_aug = use_masked_aug

        self.epoch_len = epoch_len
        self.random_frames = random_frames
        self.mask_threshold = 0.55
        self.age_threshold = 18
        # self.scale_threshold = 0.85

        # Transformsf
        if self.augment_color:
            self.aug = A.Compose(
                [A.ColorJitter(hue=0.02, p=0.8)],
                additional_targets={f'image{k}': 'image' for k in range(1, 2)})
            self.flip = A.Compose(
                [A.HorizontalFlip(p=0.5)
                 ],
                # additional_targets={f'image{k}': 'image' for k in range(1, num_source_frames + num_target_frames)})
                additional_targets={'image1': 'image', 'mask': 'image', 'mask1':'image'})

        if self.augment_rotate:
            self.aug_r = A.Compose(
                [A.SafeRotate(limit=20, p=0.99, border_mode=0, value=0),
                 ],
                additional_targets={'mask': 'image'})
            if self.augment_flip:
                self.aug_r = A.Compose(
                    [A.SafeRotate(limit=20, p=0.99, border_mode=0, value=0),
                    #  A.HorizontalFlip(p=0.1)
                     ],
                    additional_targets={'mask': 'image'})
                
        # if self.augment_rotate:
        #     self.aug_r = A.Compose(
        #         [A.SafeRotate(limit=0, p=0.5),
        #          ],
        #         additional_targets={'mask': 'image'})
        #     if self.augment_flip:
        #         self.aug_r = A.Compose(
        #             [A.SafeRotate(limit=0, p=0.5),
        #              # A.HorizontalFlip(p=0.1)
        #              ],
        #             additional_targets={'mask': 'image'})
                
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

    def __getitem__(self, index):

        # random.seed(23)

        data_dict = {
            'image': [],
            'mask':[],
            'keypoints': [],
            'face_scale': [],
            'size':[],
            'crop_box':[]
            }

        if self.augment_flip:
            a = self.filtered_indexes_pathes.shape[0] - index
            i = max(a, index)
            add_index = int(torch.randint(0, i, (1,))[0])
            index = index + add_index if a > index else index - add_index

        file = self.img_pathes_sorted[self.filtered_indexes_pathes[index]].split('/')[-1].split('.')[0]
        img_path = self.main_folder + f'/{self.img_pathes}/{file}.png'
        mask_path = self.main_folder + f'/{self.masks_pathes}/{file}.png'
        kp_path = self.main_folder + f'/{self.kp_pathes}/{file}.pkl'




        image = Image.open(img_path).convert('RGB')
        self.size, _ = image.size
        mask = Image.open(mask_path)

        mr = torch.mean(self.to_tensor(mask))
        # print(mr, ne, index, mask_path)

        with open(kp_path, 'rb') as handle:
            img_dict = pickle.load(handle)

        for i in range(2):  # 1 source and 1 target
            data_dict['image'].append(image)
            data_dict['mask'].append(mask)

            try:
                data_dict['keypoints'].append(img_dict['keypoints'])
                data_dict['face_scale'].append(img_dict['face_scale'])
                data_dict['size'].append(img_dict['size'])
            except:
                data_dict['keypoints'].append(np.zeros((68, 3)))
                print('Did not find keypoints')

        # if mr > self.mask_threshold:
        #     raise ValueError(f'Too big mask {mr, ne, index} - person too close, or, even more likely, several persons')
        #
        # try:
        #     ages = img_dict['ages']
        # except:
        #     raise ValueError('No age key')
        #
        # if len(ages)>1:
        #     raise ValueError('More then 1 person on the photo')
        #
        # if ages[0]<self.age_threshold:
        #     raise ValueError('Child photo')

            #     ne=0
            #     # if img_dict['face_scale']<self.scale_threshold:
            #     #     raise ValueError('Too little scale')
            # except Exception as e:
            #     ne+=1
            #     # print(e)
            #     # raise ValueError
            #     index = random.randrange(0, len(self.img_pathes))
            #     # print(index, ne)
            #     # index = min(index-1, 0)
            #     if ne>100:
            #         print(f'Error in ffhq dataloader: {e}, index: {index}')
            #         raise ValueError





        # Geometric augmentations and resize

        data_dict['image'] = [np.asarray(img).copy() for img in data_dict['image']]
        data_dict['mask'] = [np.asarray(m).copy() for m in data_dict['mask']]




        # Augment color
        if self.augment_color:
            imgs_dict = {(f'image{k}' if k > 0 else 'image'): img for k, img in enumerate(data_dict['image'])}
            data_dict['image'] = list(self.aug(**imgs_dict).values())
            imgs_mask_dict = {'image': data_dict['image'][0], 'image1': data_dict['image'][1], 'mask': data_dict['mask'][0], 'mask1': data_dict['mask'][1]}
            flipped = self.flip(**imgs_mask_dict)
            data_dict['image'] = [flipped['image'], flipped['image1']]
            data_dict['mask'] = [flipped['mask'], flipped['mask1']]


        # Augment rotate
        if self.augment_rotate:
            for k in range(len(data_dict['image'])):
                rotated = self.aug_r(image=data_dict['image'][k], mask=data_dict['mask'][k])
                data_dict['image'][k] = rotated['image']
                data_dict['mask'][k] = rotated['mask']

        data_dict['image'] = [Image.fromarray(img).copy() for img in data_dict['image']]
        data_dict['mask'] = [Image.fromarray(m).copy() for m in data_dict['mask']]
        data_dict = self.preprocess_data(data_dict)
        data_dict['image'] = [np.asarray(img).copy() for img in data_dict['image']]
        data_dict['mask'] = [np.asarray(m).copy() for m in data_dict['mask']]

        if self.use_masked_aug:
            data_dict['masked_face'] = [np.where(np.expand_dims(np.asarray(m).copy(), -1)>250, np.asarray(img).copy(), 0).astype(np.uint8) for img, m in zip(data_dict['image'], data_dict['mask'])]


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
        crop_box = torch.FloatTensor(data_dict['crop_box'])[:, None]

        output_data_dict = {
            'source_img': imgs[0].unsqueeze(0),
            'source_mask': masks[0].unsqueeze(0),
            'source_keypoints': keypoints[0].unsqueeze(0),
            'source_crop_box': crop_box[0].unsqueeze(0),


            'target_img': imgs[1].unsqueeze(0),
            'target_mask': masks[1].unsqueeze(0),
            'target_keypoints': keypoints[1].unsqueeze(0),
            'target_crop_box': crop_box[1].unsqueeze(0)}

        if self.output_aug_warp:
            output_data_dict['source_warp_aug'] = warp_aug[0].unsqueeze(0)
            self.aug_d = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.2)

            output_data_dict['target_warp_aug'] = self.aug_d(warp_aug[1].unsqueeze(0))
        done = True
        ne = 0
        return output_data_dict

    def preprocess_data(self, data_dict):
        MIN_SCALE = 0.75
        # MIN_SCALE = 0.95


        for i in range(len(data_dict['image'])):
            image = data_dict['image'][i]
            mask = data_dict['mask'][i]
            # size = data_dict['size'][i]
            size = int(image.size[0])
            keypoints = data_dict['keypoints'][i]*size
            face_scale = data_dict['face_scale'][i]


            if self.augment_geometric and face_scale >= MIN_SCALE:
                # Random sized crop
                min_scale = MIN_SCALE / face_scale
                # seed = (random.random() + 1)/4  # чтобы максимаьный скейл был поменьше всё же
                seed = random.random()
                scale = seed * (1 - min_scale) + min_scale
                scale = max(min(1, scale), 0)
                translate_x = random.random() * (1 - scale)
                translate_y = random.random() * (1 - scale)

            else:
                translate_x = 0
                translate_y = 0
                scale = 1

            crop_box = (size * translate_x,
                        size * translate_y,
                        size * (translate_x + scale),
                        size * (translate_y + scale))


            # print(image.size, size, translate_x, scale, translate_x + scale, size * (translate_y + scale))

            size_box = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])


            keypoints[..., 0] = (keypoints[..., 0] - crop_box[0]) / size_box[0] - 0.5
            keypoints[..., 1] = (keypoints[..., 1] - crop_box[1]) / size_box[1] - 0.5

            try:
                keypoints[..., 2] = keypoints[..., 2] / (size_box[0] + size_box[1]) * 2
            except:
                keypoints = np.concatenate((keypoints, keypoints[..., 0:1]*0), axis=1)
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
            return len(self.filtered_indexes_pathes)
        else:
            return self.epoch_len


class DataModule(object):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser, *, use_argument_group=True):
        if use_argument_group:
            parser = parent_parser.add_argument_group("dataset")
            parser_out = parent_parser
        else:
            parser = ArgumentParser(parents=[parent_parser], add_help=False)
            parser_out = parser

        parser.add_argument('--ffhq_batch_size', default=2, type=int)
        parser.add_argument('--ffhq_test_batch_size', default=2, type=int)
        parser.add_argument('--ffhq_num_workers', default=16, type=int)
        parser.add_argument('--main_path', default='/fsx/behavioural_computing_data/face_generation_data/FFHQ_wild/filtered', type=str)
        parser.add_argument('--filtered_indexes', default='indexes_train_055_16.npy', type=str)
        parser.add_argument('--test_filtered_indexes', default='indexes_test_055_16.npy', type=str)

        parser.add_argument('--ffhq_data_root', default='images_from_wild', type=str)
        parser.add_argument('--ffhq_masks_root', default='masks_from_wild', type=str)
        parser.add_argument('--ffhq_kp_root', default='keypoints_from_wild', type=str)

        parser.add_argument('--ffhq_test_data_root', default='images_from_wild_test', type=str)
        parser.add_argument('--ffhq_test_masks_root', default='masks_from_wild_test', type=str)
        parser.add_argument('--ffhq_test_kp_root', default='keypoints_from_wild_test', type=str)
        parser.add_argument('--ffhq_num_source_frames', default=1, type=int)
        parser.add_argument('--ffhq_num_target_frames', default=1, type=int)
        parser.add_argument('--ffhq_image_size', default=512, type=int)
        parser.add_argument('--ffhq_augment_geometric_train', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--ffhq_augment_color_train', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--ffhq_output_aug_warp', default='True', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--ffhq_use_masked_aug', default='False', type=args_utils.str2bool, choices=[True, False])
        parser.add_argument('--ffhq_aug_warp_size', default=512, type=int)



        # These parameters can be used for debug
        parser.add_argument('--ffhq_train_epoch_len', default=-1, type=int)
        parser.add_argument('--ffhq_test_epoch_len', default=-1, type=int)

        return parser_out

    def __init__(self, args):
        super(DataModule, self).__init__()
        self.ddp = args.num_gpus > 1
        self.batch_size = args.ffhq_batch_size
        self.test_batch_size = args.ffhq_test_batch_size
        self.num_workers = args.ffhq_num_workers
        self.data_root = args.ffhq_data_root
        self.masks_root = args.ffhq_masks_root
        self.kp_root = args.ffhq_kp_root
        self.test_kp_root = args.ffhq_test_kp_root
        self.test_data_root = args.ffhq_test_data_root
        self.test_masks_root = args.ffhq_test_masks_root
        self.num_source_frames = args.ffhq_num_source_frames
        self.num_target_frames = args.ffhq_num_target_frames
        self.image_size = args.ffhq_image_size
        self.augment_geometric_train = args.ffhq_augment_geometric_train
        self.augment_color_train = args.ffhq_augment_color_train
        self.output_aug_warp = args.ffhq_output_aug_warp
        self.use_masked_aug = args.ffhq_use_masked_aug
        self.aug_warp_size = args.ffhq_aug_warp_size
        self.train_epoch_len = args.ffhq_train_epoch_len
        self.test_epoch_len = args.ffhq_test_epoch_len
        self.filtered_indexes = args.filtered_indexes
        self.test_filtered_indexes = args.test_filtered_indexes
        self.main_path = args.main_path
        self.keys = {'test': [], 'train': []}
        self.splits = {'test': [], 'train': []}

    def train_dataloader(self):
        train_dataset = FFHQ_Crops_Dataset(self.filtered_indexes,
                                           self.main_path,
                                           self.data_root,
                                           self.masks_root,
                                           self.kp_root,
                                           self.image_size,
                                           self.augment_geometric_train,
                                           self.augment_color_train,
                                           self.output_aug_warp,
                                           self.use_masked_aug,
                                           self.aug_warp_size,
                                           self.train_epoch_len,
                                           random_frames=True,
                                           augment_rotate=True,
                                           augment_flip=True
                                           )


        shuffle = True
        sampler = None
        if self.ddp:
            shuffle = False
            sampler = data.distributed.DistributedSampler(train_dataset, shuffle = False, seed=1)


        return data.DataLoader(train_dataset,
        					   batch_size=self.batch_size,
        					   num_workers=self.num_workers,
        					   pin_memory=True,
        					   shuffle=shuffle,
                               sampler=sampler)

    def test_dataloader(self):
        test_dataset = FFHQ_Crops_Dataset(  self.test_filtered_indexes,
                                            self.main_path,
                                            self.test_data_root,
                                            self.test_masks_root,
                                            self.test_kp_root,
                                            self.image_size,
                                            epoch_len=self.test_epoch_len,
                                            random_frames=True,
                                            augment_rotate=True)

        sampler = None
        if self.ddp:
            sampler = data.distributed.DistributedSampler(test_dataset, shuffle=True, seed=1)

        return data.DataLoader(test_dataset,
        					   batch_size=self.test_batch_size,
        					   num_workers=self.num_workers,
        					   pin_memory=True,
                               sampler=sampler)