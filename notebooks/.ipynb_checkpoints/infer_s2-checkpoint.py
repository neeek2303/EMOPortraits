import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import os
import pathlib
import numpy as np
import importlib
import math
from scipy import linalg
import apex
import utils.args as args_utils
from utils import spectral_norm, stats_calc
from datasets.voxceleb2hq_pairs import LMDBDataset
from repos.MODNet.src.models.modnet import MODNet
import sys
sys.path.append('/fsx/nikitadrobyshev/EmoPortraits')
from networks.volumetric_avatar import FaceParsing
from torch.nn.modules.module import _addindent
import mediapipe as mp
from PIL import Image

mp_face_detection = mp.solutions.face_detection

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

class InferenceWrapper(nn.Module):
    def __init__(self, experiment_name,  which_epoch='latest', model_file_name='', use_gpu=True, num_gpus = 1, fixed_bounding_box=False, project_dir='./',
                 torch_home='', debug=False,  print_model=False, args_overwrite={}, pose_momentum=0.5, experiment_name_s1=None, model_file_name_s1=None, cloth=False):
        super(InferenceWrapper, self).__init__()
        self.use_gpu = use_gpu
        self.debug = debug
        self.num_gpus = num_gpus

        self.modnet_pass =  f'{project_dir}/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
        self.cloth=cloth
        # Get a config for the network
        args_path = pathlib.Path(project_dir) / 'logs_s2' / experiment_name / 'args.txt'

        self.args = args_utils.parse_args(args_path)
        # Add args from args_overwrite dict that overwrite the default ones
        self.args.project_dir = project_dir
        if args_overwrite is not None:
            for k, v in args_overwrite.items():
                setattr(self.args, k, v)

        
        self.face_idt = FaceParsing(None, 'cuda')

        if torch_home:
            os.environ['TORCH_HOME'] = torch_home


        if self.num_gpus > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.random_seed)

        self.check_grads = self.args.check_grads_of_every_loss
        self.print_model = print_model
        # Set distributed training options
        if self.num_gpus <= 1:
            self.rank = 0

        elif self.num_gpus > 1 and self.num_gpus <= 8:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.rank)

        elif self.num_gpus > 8:
            raise


        
        # Initialize model
        self.model_two = importlib.import_module(f'models.stage_2.base.{self.args.model_name}_two').Model(self.args, training=False)

        if self.use_gpu:
            self.model_two.cuda()

        if self.rank == 0 and self.print_model:
            print('=================================================================================================')
            print(self.model_two)
            # ms = torch_summarize(self.model)

        # Load pre-trained weights
        if experiment_name_s1 and model_file_name_s1:
            self.model_checkpoint = pathlib.Path(project_dir) / 'logs' / experiment_name_s1 / 'checkpoints' / model_file_name_s1
        else:
            self.model_checkpoint = self.args.model_checkpoint



        self.model_checkpoint_s2 = pathlib.Path(project_dir) / 'logs_s2' / experiment_name / 'checkpoints' / model_file_name
        if self.rank == 0:
            print(f'Loading model from {self.model_checkpoint_s2}')
        self.model_dict_s2 = torch.load(self.model_checkpoint_s2, map_location='cpu')
        self.model_two.load_state_dict(self.model_dict_s2, strict=False)


        # Initialize distributed training
        if self.num_gpus > 1:
            self.model_two = apex.parallel.convert_syncbn_model(self.model_two)
            self.model_two = apex.parallel.DistributedDataParallel(self.model_two)

        self.model_two.eval()

        self.modnet = MODNet(backbone_pretrained=False)

        if self.num_gpus > 0:
            self.modnet = nn.DataParallel(self.modnet).cuda()

        if self.use_gpu:
            self.modnet = self.modnet.cuda()

        self.modnet.load_state_dict(torch.load(self.modnet_pass))
        self.modnet.eval()

        # Face detection is required as pre-processing
        device = 'cuda' if use_gpu else 'cpu'
        self.device = device
        face_detector = 'sfd'
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=False)

        # Face tracking and bounding box smoothing parameters
        self.fixed_bounding_box = fixed_bounding_box  # no tracking is performed, first bounding box in driver is used for all frames
        self.momentum = 0.01  # if bounding box is not fixed, it is updated with momentum
        self.center = None
        self.size = None

        # Head pose smoother
        self.pose_momentum = pose_momentum
        self.theta = None

        # Head normalization params
        self.norm_momentum = 0.1
        self.delta_yaw = None
        self.delta_pitch = None

        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()
        self.resize_warp = self.args.warp_output_size != self.args.gen_latent_texture_size
        self.use_seg = self.args.use_seg

    @torch.no_grad()
    def calculate_standing_stats(self, data_root, num_iters):
        self.identity_embedder.train().apply(stats_calc.stats_calculation)
        self.pose_embedder.train().apply(stats_calc.stats_calculation)
        self.generator.train().apply(stats_calc.stats_calculation)

        # Initialize train dataset
        dataset = LMDBDataset(
            data_root,
            'train',
            self.args.num_source_frames,
            self.args.num_target_frames,
            self.args.image_size,
            False)

        dataset.names = dataset.names[:self.args.batch_size * num_iters]

        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=self.args.num_workers_per_process)

        for data_dict in dataloader:
            source_img_crop = data_dict['source_img']
            driver_img_crop = data_dict['target_img']

            source_img_crop = source_img_crop.view(-1, *source_img_crop.shape[2:])
            driver_img_crop = driver_img_crop.view(-1, *driver_img_crop.shape[2:])

            if self.use_gpu:
                source_img_crop = source_img_crop.cuda()
                driver_img_crop = driver_img_crop.cuda()

            idt_embed = self.identity_embedder.forward_image(source_img_crop)

            # During training, pose embedder accepts concatenated data, so we need to imitate it during stats calculation
            img_crop = torch.cat([source_img_crop, driver_img_crop])
            pose_embed, pred_theta = self.pose_embedder.forward_image(img_crop)

            source_pose_embed, driver_pose_embed = pose_embed.split(
                [source_img_crop.shape[0], driver_img_crop.shape[0]])
            pred_source_theta, pred_driver_theta = pred_theta.split(
                [source_img_crop.shape[0], driver_img_crop.shape[0]])

            latent_texture, embed_dict = self.generator.forward_source(source_img_crop, idt_embed, source_pose_embed,
                                                                       pred_source_theta)
            pred_target_img = self.generator.forward_driver(idt_embed, driver_pose_embed, embed_dict, pred_source_theta,
                                                            pred_driver_theta, latent_texture)


    def convert_to_tensor(self, image):
        if isinstance(image, list):
            image_tensor = [self.to_tensor(img) for img in image]
            image_tensor = torch.stack(image_tensor)  # all images have to be the same size
        else:
            image_tensor = self.to_tensor(image)

        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor[None]

        if self.use_gpu:
            image_tensor = image_tensor.cuda()

        return image_tensor

    @staticmethod
    def remove_overflow(center, size, w, h):
        bbox = np.asarray([center[0] - size / 2, center[1] - size / 2, center[0] + size / 2, center[1] + size / 2])

        shift_l = 0 if bbox[0] >= 0 else -bbox[0]
        shift_u = 0 if bbox[1] >= 0 else -bbox[1]
        shift_r = 0 if bbox[2] <= w else bbox[2] - w
        shift_d = 0 if bbox[3] <= h else bbox[3] - h

        shift = max(shift_l, shift_u, shift_r, shift_d)

        bbox[[0, 1]] += shift
        bbox[[2, 3]] -= shift

        center = np.asarray([bbox[[0, 2]].mean(), bbox[[1, 3]].mean()]).astype(int)
        size_overflow = int((bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 2)
        size_overflow = size_overflow - size_overflow % 2

        return size_overflow

    def crop_image_old(self, image, faces, use_smoothed_crop=False):
        imgs_crop = []

        for b, face in enumerate(faces):

            assert face is not None, 'Face not found!'

            center = np.asarray([(face[2] + face[0]) // 2, (face[3] + face[1]) // 2])
            size = face[2] - face[0] + face[3] - face[1]

            if use_smoothed_crop:
                if self.center is None:
                    self.center = center
                    self.size = size

                elif not self.fixed_bounding_box:
                    self.center = center * self.momentum + self.center * (1 - self.momentum)
                    self.size = size * self.momentum + self.size * (1 - self.momentum)

                center = self.center
                size = self.size

            center = center.round().astype(int)
            size = int(round(size))
            size = size - size % 2
            size = self.remove_overflow(center, size, image.shape[3], image.shape[2])

            img_crop = image[b, :, center[1] - size // 2: center[1] + size // 2,
                       center[0] - size // 2: center[0] + size // 2]
            img_crop = F.interpolate(img_crop[None], size=(self.args.image_size, self.args.image_size), mode='bicubic')

            imgs_crop += [img_crop]

        imgs_crop = torch.cat(imgs_crop)

        return imgs_crop

    def crop_image(self, image, faces, use_smoothed_crop=False):
        imgs_crop = []
        face_check = np.ones(len(image), dtype=bool)
        face_scale_stats = []

        for b, face in enumerate(faces):

            if face is None:
                face_check[b] = False
                imgs_crop.append(torch.zeros((1, 3, self.args.image_size, self.args.image_size)))
                face_scale_stats.append(0)
                continue

            center = np.asarray([(face[2] + face[0]) // 2, (face[3] + face[1]) // 2])
            size = (face[2] - face[0] + face[3] - face[1])

            if use_smoothed_crop:
                if self.center is None:
                    self.center = center
                    self.size = size

                elif not self.fixed_bounding_box:
                    self.center = center * self.momentum + self.center * (1 - self.momentum)
                    self.size = size * self.momentum + self.size * (1 - self.momentum)

                center = self.center
                size = self.size

            center = center.round().astype(int)
            size = int(round(size))
            size = size - size % 2

            if isinstance(image, list):
                size_overflow = self.remove_overflow(center, size, image[b].shape[2], image[b].shape[1])
                face_scale = size_overflow / size
                size = size_overflow
                img_crop = image[b][:, center[1] - size // 2: center[1] + size // 2,
                           center[0] - size // 2: center[0] + size // 2]
            else:
                size_overflow = self.remove_overflow(center, size, image.shape[3], image.shape[2])
                face_scale = size_overflow / size
                size = size_overflow
                img_crop = image[b, :, center[1] - size // 2: center[1] + size // 2,
                           center[0] - size // 2: center[0] + size // 2]

            img_crop = F.interpolate(img_crop[None], size=(self.args.image_size, self.args.image_size), mode='bicubic')
            imgs_crop.append(img_crop)
            face_scale_stats.append(face_scale)

        imgs_crop = torch.cat(imgs_crop).clip(0, 1)

        return imgs_crop, face_check, face_scale_stats



    @torch.no_grad()
    def forward(self, img, cloth=False):
            data_dict = {}
            
            # Stage 2
            #==========================================================================================================
            data_dict['pred_target_mask'] = self.get_mask(img)
            data_dict['pred_target_img'] = img
            resize_n = lambda img: F.interpolate(img, mode='nearest', size=(self.args.output_size_s2, self.args.output_size_s2))
            resize = lambda img: F.interpolate(img, mode='bilinear',size=(self.args.output_size_s2, self.args.output_size_s2), align_corners=False)
            data_dict['resized_pred_target_img'] = resize(data_dict['pred_target_img'])
            data_dict['resized_pred_target_mask'] = self.get_mask(data_dict['resized_pred_target_img'])
            face_mask_source, _, _, _ = self.face_idt.forward(data_dict['resized_pred_target_img'])

            if not self.cloth:
                data_dict['resized_pred_target_face_mask'] = data_dict['resized_pred_target_mask'] * face_mask_source
            else:
                data_dict['resized_pred_target_face_mask'] = data_dict['resized_pred_target_mask']


            aligned_target_volume = self.model_two.local_encoder(data_dict['resized_pred_target_img'] * data_dict['resized_pred_target_mask'])
            data_dict['pred_target_add'], _, _, _ = self.model_two.decoder(None, None, aligned_target_volume, 
                                                                           False, pred_feat=None)
            data_dict['pred_target_add'] = data_dict['pred_target_add']*data_dict['resized_pred_target_face_mask']
            data_dict['pred_target_img_ffhq'] = data_dict['resized_pred_target_img'] + data_dict['pred_target_add']
            data_dict['pred_target_img_ffhq'].clamp_(max=1, min=0)



            pred_target_img = img.detach().cpu().clamp(0, 1)
            pred_target_img = [self.to_image(img) for img in pred_target_img]

            pred_target_img_resized = [self.to_image(img) for img in data_dict['resized_pred_target_img']]
            pred_target_img_ffhq = [self.to_image(img) for img in data_dict['pred_target_img_ffhq']]


            return pred_target_img, pred_target_img_resized, pred_target_img_ffhq, data_dict['pred_target_mask'].detach().cpu().clamp(0, 1)




    def get_mask(self, img):

        im_transform = transforms.Compose(
            [

                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        im = im_transform(img)
        ref_size = 512
        # add mini-batch dim

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = self.modnet(im.cuda(), True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

        return matte

    def get_mixing_theta(self, source_theta, target_theta):
        source_theta = source_theta[:, :3, :]
        target_theta = target_theta[:, :3, :]
        N = 1
        B = source_theta.shape[0] // N
        T = target_theta.shape[0] // B

        source_theta_ = np.stack([np.eye(4) for i in range(B)])
        target_theta_ = np.stack([np.eye(4) for i in range(B * T)])

        source_theta = source_theta.view(B, N, *source_theta.shape[1:])[:, 0]  # take theta from the first source image
        target_theta = target_theta.view(B, T, 3, 4).roll(1, dims=0).view(B * T, 3, 4)  # shuffle target poses

        source_theta_[:, :3, :] = source_theta.detach().cpu().numpy()
        target_theta_[:, :3, :] = target_theta.detach().cpu().numpy()

        # Extract target translation
        target_translation = np.stack([np.eye(4) for i in range(B * T)])
        target_translation[:, :3, 3] = target_theta_[:, :3, 3]

        # Extract linear components
        source_linear_comp = source_theta_.copy()
        source_linear_comp[:, :3, 3] = 0

        target_linear_comp = target_theta_.copy()
        target_linear_comp[:, :3, 3] = 0

        pred_mixing_theta = []
        for b in range(B):
            # Sometimes the decomposition is not possible, hense try-except blocks
            try:
                source_rotation, source_stretch = linalg.polar(source_linear_comp[b])
            except:
                pred_mixing_theta += [target_theta_[b * T + t] for t in range(T)]
            else:
                for t in range(T):
                    try:
                        target_rotation, target_stretch = linalg.polar(target_linear_comp[b * T + t])
                    except:
                        pred_mixing_theta.append(source_stretch)
                    else:
                        pred_mixing_theta.append(target_translation[b * T + t] @ target_rotation @ source_stretch)
#                         pred_mixing_theta.append(source_stretch * target_stretch.mean() / source_stretch.mean() @ target_rotation @ target_translation[b * T + t])
        pred_mixing_theta = np.stack(pred_mixing_theta)

        return torch.from_numpy(pred_mixing_theta)[:, :3].type(source_theta.type()).to(source_theta.device)



    # def get_mixing_theta(self, source_theta, target_theta):
    #     source_theta = source_theta[:, :3, :]
    #     target_theta = target_theta[:, :3, :]
    #     N = 1
    #     B = source_theta.shape[0] // N
    #     T = target_theta.shape[0] // B
    #
    #     source_theta_ = np.stack([np.eye(4) for i in range(B)])
    #     target_theta_ = np.stack([np.eye(4) for i in range(B * T)])
    #
    #     source_theta = source_theta.view(B, N, *source_theta.shape[1:])[:, 0]  # take theta from the first source image
    #     target_theta = target_theta.view(B, T, 3, 4).roll(1, dims=0).view(B * T, 3, 4)  # shuffle target poses
    #
    #     source_theta_[:, :3, :] = source_theta.detach().cpu().numpy()
    #     target_theta_[:, :3, :] = target_theta.detach().cpu().numpy()
    #
    #     # Extract target translation
    #     target_translation = np.stack([np.eye(4) for i in range(B * T)])
    #     target_translation[:, :3, 3] = target_theta_[:, :3, 3]
    #
    #     # Extract linear components
    #     source_linear_comp = source_theta_.copy()
    #     source_linear_comp[:, :3, 3] = 0
    #
    #     target_linear_comp = target_theta_.copy()
    #     target_linear_comp[:, :3, 3] = 0
    #
    #     pred_mixing_theta = []
    #     for b in range(B):
    #         # Sometimes the decomposition is not possible, hense try-except blocks
    #         try:
    #             source_rotation, source_stretch = linalg.polar(source_linear_comp[b])
    #             source_scale = np.diag(source_stretch)
    #             source_stretch = np.diag(source_scale)
    #             source_scale = source_scale[:3]/source_scale[:3].mean()
    #             # print(source_stretch)
    #         except:
    #             pred_mixing_theta += [target_theta_[b * T + t] for t in range(T)]
    #         else:
    #             for t in range(T):
    #                 try:
    #                     target_rotation, target_stretch = linalg.polar(target_linear_comp[b * T + t])
    #                     target_scale = np.diag(target_stretch)
    #                     target_scale[:3] = source_scale*target_scale[:3].mean()
    #                     target_stretch = np.diag(target_scale)
    #
    #
    #                 except:
    #                     pred_mixing_theta.append(source_stretch)
    #                 else:
    #                     pred_mixing_theta.append(target_translation[b * T + t] @ target_rotation @ target_stretch)
    #
    #     pred_mixing_theta = np.stack(pred_mixing_theta)
    #
    #     return torch.from_numpy(pred_mixing_theta)[:, :3].type(source_theta.type()).to(source_theta.device)