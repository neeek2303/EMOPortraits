import torch
from torch import nn
import copy
import argparse
import os
import pathlib
import importlib
import ssl
import time
import sys
from tqdm import tqdm
import functools
from PIL import Image
from utils import args as args_utils
from utils.logger_wandb import Logger
import warnings
warnings.filterwarnings("ignore")
from torchvision import transforms
import wandb
import os

os.environ["WANDB_SILENT"] = "True"

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp


# 



class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        # Initialize and apply general options

        if args.use_torch_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
        else:
            import apex
            from apex import amp


        ssl._create_default_https_context = ssl._create_unverified_context
        torch.manual_seed(args.random_seed)
        if args.num_gpus > 0:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.random_seed)

        self.args = args
        self.custom_test = args.custom_test
        self.check_grads = args.check_grads_of_every_loss
        self.to_tensor = transforms.ToTensor()
        # Set distributed training options
        if args.num_gpus <= 1:
            self.rank = 0

        elif args.num_gpus > 1 and args.num_gpus <= 8:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.rank)

        elif args.num_gpus > 8:
            raise

        # Prepare experiment directories and save options
        self.project_dir = pathlib.Path(args.project_dir)
        self.experiment_dir = self.project_dir / 'logs' / args.experiment_name
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        os.makedirs(self.checkpoints_dir, exist_ok=True)


        if self.rank == 0:
            with open(self.experiment_dir / 'args.txt', 'wt') as args_file:
                for k, v in sorted(vars(args).items()):
                    args_file.write('%s: %s\n' % (str(k), str(v)))

        self.exp_dir = None
        if self.args.save_exp_vectors:
            self.exp_dir = self.experiment_dir / 'expression_vectors'
            os.makedirs(self.exp_dir, exist_ok=True)

        # Initialize model
        self.model = importlib.import_module(f'models.stage_1.{args.model_type}.{args.model_name}').Model(args, rank=self.rank, exp_dir=self.exp_dir)

        if args.num_gpus > 0:
            self.model.cuda()

        if self.rank == 0 and self.args.print_model:
            print(self.model)

        # Load pre-trained weights
        if args.model_checkpoint:
            if self.rank == 0:
                print(f'Loading model from {args.model_checkpoint}')
            self.model.load_state_dict(torch.load(args.model_checkpoint, map_location='cpu'), strict=False)

        # Initialize optimizers and schedulers
        self.opts = self.model.configure_optimizers()



        # # Initialize mixed precision
        # if args.use_amp:
        #     self.model, self.opts = amp.initialize(self.model, self.opts, opt_level=args.amp_opt_level, num_losses=len(self.opts))

        # Initialize dataloaders
        data_module = importlib.import_module(f'datasets.{args.dataset_name_test}').DataModule(args)
        self.test_dataloader = data_module.test_dataloader()

        data_module = importlib.import_module(f'datasets.{args.dataset_name}').DataModule(args)
        self.train_dataloader = data_module.train_dataloader()

        if self.args.use_sec_dataset:
            # Initialize dataloaders
            self.second_iter_count=0

            data_module_sec = importlib.import_module(f'datasets.{args.dataset_name_test_sec}').DataModule(args)
            self.test_dataloader_sec = data_module_sec.test_dataloader()
            
            
            data_module_sec = importlib.import_module(f'datasets.{args.dataset_name_sec}').DataModule(args)
            self.train_dataloader_sec = data_module_sec.train_dataloader()

            if self.args.mead_as_second_every>0:

                data_module_mead = importlib.import_module(f'datasets.{args.dataset_name_test_mead}').DataModule(args)
                self.test_dataloader_mead = data_module_mead.test_dataloader()
                

                data_module_mead = importlib.import_module(f'datasets.{args.dataset_name_mead}').DataModule(args)
                self.train_dataloader_mead = data_module_mead.train_dataloader()

            


        self.shds, self.shd_max_iters = self.model.configure_schedulers(self.opts, epochs=self.args.max_epochs, steps_per_epoch=len(self.train_dataloader))

        # Initialize logging
        self.logger = Logger(args, self.experiment_dir, self.rank, self.model, project_name = self.args.project_name, entity=self.args.entity)

        # Load pre-trained optimizers and schedulers
        if args.trainer_checkpoint:
            if self.rank == 0:
                print(f'Loading trainer from {args.trainer_checkpoint}')
            trainer_checkpoint = torch.load(args.trainer_checkpoint, map_location='cpu')

            for i, opt in enumerate(self.opts):
                try:
                    opt.load_state_dict(trainer_checkpoint[f'opt_{i}'])
                except Exception as e:
                    print(f'Was not able to load opt number {i}')

            if len(self.shds):
                for i, shd in enumerate(self.shds):
                    try:
                        shd.load_state_dict(trainer_checkpoint[f'shd_{i}'])
                    except Exception as e:
                        print(f'Was not able to load sheduler number {i}')

            # if args.use_amp and 'amp' in trainer_checkpoint.keys():
            #     amp.load_state_dict(trainer_checkpoint['amp'])

            self.logger.load_state_dict(trainer_checkpoint['logger'])

        if self.rank == 0:
            print(f'Optimizing networks: {self.model.opt_net_names}')
            print(f'Optimizing tensors: {self.model.opt_tensor_names}')
            print(f'Optimizing discriminators: {self.model.opt_dis_names}')
            for n, p in self.model.net_param_dict.items():
                print(f'Number of perameters in {n}: {p}')
            

        # Initialize distributed training
        if args.num_gpus > 1:
            if self.args.use_torch_ddp:
                self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
            else:
                self.model = apex.parallel.convert_syncbn_model(self.model)
                self.model = apex.parallel.DistributedDataParallel(self.model)
            

    @staticmethod
    def get_lr(opt, use_gpu):
        for param_group in opt.param_groups:
            lr = param_group['lr']
            lr = torch.FloatTensor([lr]).mean()
            if use_gpu:
                lr = lr.cuda()

            return lr

    def train(self):


        for n in range(self.logger.epoch, self.args.max_epochs):
            
            if self.args.num_gpus>1:
                self.train_dataloader.sampler.set_epoch(n)
                if self.args.use_sec_dataset:
                    self.train_dataloader_sec.sampler.set_epoch(n)
                    if self.args.mead_as_second_every>0:
                        self.train_dataloader_mead.sampler.set_epoch(n)

            if self.rank == 0:
                train_data_iterator = tqdm(self.train_dataloader)
                test_data_iterator = tqdm(self.test_dataloader)
            else:
                train_data_iterator = self.train_dataloader
                test_data_iterator = self.test_dataloader
            
            if self.args.use_sec_dataset:
                self.train_data_iterator_sec = iter(self.train_dataloader_sec)
                if self.args.mead_as_second_every>0:
                        self.train_data_iterator_mead = iter(self.train_dataloader_mead)

            self.len_test = len(test_data_iterator)

            # Train
            self.model.train()
            to_image = transforms.ToPILImage()
            for i, data_dict in enumerate(train_data_iterator):

                # if i%self.args.sec_dataset_every == 0:
                #     data_dict = next(self.train_data_iterator_sec)
                if self.args.use_sec_dataset:
                    if self.args.mead_as_second_every>0 and self.second_iter_count%self.args.mead_as_second_every: 
                        curr_iter = self.train_data_iterator_mead
                    else:
                        curr_iter = self.train_data_iterator_sec

                    
                    if self.args.sec_dataset_every%2!=0:
                        # if i%self.args.sec_dataset_every == 0 or (i+1)%self.args.sec_dataset_every == 0 or (i+2)%self.args.sec_dataset_every == 0 or (i+3)%self.args.sec_dataset_every == 0 or (i+4)%self.args.sec_dataset_every == 0:
                        if i%self.args.sec_dataset_every == 0 or (i+1)%self.args.sec_dataset_every == 0:    
                            data_dict_ = next(curr_iter)
                            data_dict = {k:torch.cat([data_dict[k][:1], data_dict_[k][1:]], dim=0) for k in data_dict.keys()}
                    else:
                        if i%(self.args.sec_dataset_every//2) == 0:
                            data_dict_ = next(curr_iter)
                            self.second_iter_count+=1
                            data_dict = {k:torch.cat([data_dict[k][:1], data_dict_[k][1:]], dim=0) for k in data_dict.keys()}
                

                # with torch.autograd.set_detect_anomaly(True):
                losses_dict, visuals = self.training_step(data_dict, self.check_grads, epoch=n, iteration=i)
                # if n==0:
                #     if i<10:
                #         losses_dict.update((x, y * 0.1) for x, y in losses_dict.items())

                if len(self.shds):
                    for i, opt in enumerate(self.opts):
                        losses_dict[f'opt_{i}_lr'] = self.get_lr(opt, self.args.num_gpus > 0)

                if self.args.normalize_losses:

                    keys_losses = list(losses_dict.keys())

                    if 'l1_eyes' in keys_losses:
                        losses_dict['l1_eyes']/=self.args.w_eyes_loss_l1/100
                    
                    if 'l1_mouth' in keys_losses:
                        losses_dict['l1_mouth']/=self.args.w_mouth_loss_l1/100

                    if 'l1_ears' in keys_losses:
                        losses_dict['l1_ears']/=self.args.w_ears_loss_l1/100
                    
                    if 'vgg19_face' in keys_losses:
                        losses_dict['vgg19_face']/=self.args.vgg19_face/4

                    if 'pull_exp' in keys_losses:
                        losses_dict['pull_exp']/=self.args.pull_exp/0.5

                    if 'push_exp' in keys_losses:
                        losses_dict['push_exp']/=self.args.push_exp/0.5
                    
                    if 'resnet18_fv_mix' in keys_losses:
                        losses_dict['resnet18_fv_mix']/=self.args.resnet18_fv_mix/35

                    if 'volumes_l1_loss' in keys_losses:
                        losses_dict['volumes_l1_loss']/=self.args.volumes_l1

                        

                        

                self.logger.log('train', losses_dict, visuals)

            torch.cuda.empty_cache()
            ########################################################################################
            #####################################  Test  ###########################################
            ########################################################################################

            time.sleep(15)
            self.model.eval()
            del  data_dict, visuals
       
            for i, data_dict in enumerate(test_data_iterator):
                
                with torch.no_grad():
                    first_batch = i == 0
                    
                    iteration = i if i!=self.len_test-1 else -1
                    
                    # Add custom images to test set
                    if self.custom_test and self.rank == 0 and first_batch:
                        size = data_dict['source_img'].shape[-1]
                        b = data_dict['source_img'].shape[0]
                        image_list = [f'{args.project_dir}/data/one.png', f'{args.project_dir}/data/ton_512.png', f'{args.project_dir}/data/two.png',
                                      f'{args.project_dir}/data/asim_512.png']
                        mask_list = [f'{args.project_dir}/data/j1_mask.png', f'{args.project_dir}/data/j1_mask.png', f'{args.project_dir}/data/j1_mask.png',
                                     f'{args.project_dir}/data/j1_mask.png']
                        image_list = image_list[:b]
                        mask_list = mask_list[:b]
                        images = []
                        masks = []
                        for im, m in zip(image_list, mask_list):
                            image = Image.open(im).convert('RGB')
                            mask = Image.open(m)
                            image = image.resize((size, size), Image.BICUBIC)
                            mask = mask.resize((size, size), Image.BICUBIC)
                            images.append(self.to_tensor(image).unsqueeze(0))
                            masks.append(torch.ones_like(self.to_tensor(mask)).unsqueeze(0)) # hack to avoid masking
                        test_data_dict = copy.deepcopy(data_dict)
                        test_data_dict['source_img'] = torch.stack((images), dim=0)
                        test_data_dict['source_mask'] = torch.stack((masks), dim=0)
                        test_data_dict['target_img'] = torch.stack((images), dim=0)
                        test_data_dict['target_mask'] = torch.stack((masks), dim=0)
                        _, _, visuals_, _ = self.model(test_data_dict, visualize=first_batch, iteration=iteration, rank = self.rank, epoch = n)
                        del test_data_dict

                        _, losses_dict, _, data_dict_ = self.model(data_dict, visualize=False, iteration=iteration, rank = self.rank, epoch = n)
                        self.logger.log('test', losses_dict)
                        try:
                            self.expl_var = data_dict_['expl_var']
                        except Exception as e:
                            pass

                    else:
                        _, losses_dict, visuals_, _ = self.model(data_dict, visualize=first_batch, iteration=iteration, rank = self.rank, epoch = n)
                        self.logger.log('test', losses_dict)

                    if first_batch:
                        visuals = visuals_ # store visuals from the first batch

            self.logger.log('test', visuals=visuals, epoch_end=True, explaining_var = self.expl_var if hasattr(self, 'expl_var') else None)
            del visuals_, visuals
            epoch = self.logger.epoch
            
            # Save checkpoints
            if self.rank == 0 and (not epoch % self.args.latest_checkpoint_freq or not epoch % self.args.checkpoint_freq):
                # Model
                if self.args.num_gpus > 1:
                    model = self.model.module
                else:
                    model = self.model

                torch.save(model.state_dict(), self.checkpoints_dir / f'{epoch:03d}_model.pth')

                # Trainer
                trainer_checkpoint = {}

                for i, opt in enumerate(self.opts):
                    trainer_checkpoint[f'opt_{i}'] = opt.state_dict()

                if len(self.shds):
                    for i, shd in enumerate(self.shds):
                        trainer_checkpoint[f'shd_{i}'] = shd.state_dict()

                # if args.use_amp:
                #     trainer_checkpoint['amp'] = amp.state_dict()

                trainer_checkpoint['logger'] = self.logger.state_dict()

                torch.save(trainer_checkpoint, self.checkpoints_dir / f'{epoch:03d}_trainer.pth')

                # Remove previous checkpoint
                prev_epoch = epoch - 1
                if epoch > 1 and prev_epoch % self.args.checkpoint_freq:
                    try:
                        os.remove(self.checkpoints_dir / f'{prev_epoch:03d}_model.pth')
                        os.remove(self.checkpoints_dir / f'{prev_epoch:03d}_trainer.pth')
                    except:
                        print('previous checkpoints not found')

            torch.cuda.empty_cache()
            time.sleep(15)

    def training_step(self, data_dict, check_grads=False, epoch=0, iteration = 0):
        output_visuals = self.logger.output_train_visuals and self.args.output_visuals
        losses_dict = {}
        visuals = torch.empty(0)

        for i, opt in enumerate(self.opts):


            if self.args.use_torch_ddp:
                data_dict_ = {k: v.clone() for k, v in data_dict.items()}
                opt.zero_grad()
                if check_grads:
                    data_dict_['source_img'] = data_dict_['source_img'].requires_grad_(True)
                    data_dict_['source_img'].retain_grad()
                
                loss, losses_dict_, visuals_, data_dict_ = self.model(data_dict_, 'train', i,
                                                                      visualize=output_visuals and i == 0,
                                                                      iteration=iteration, rank = self.rank,
                                                                      epoch = epoch)
                losses_dict.update(losses_dict_)

                if i == 0 and visuals_ is not None:
                    visuals.data = visuals_.data
                
                if i==0:
                    data_dict['pred_target_img'] = data_dict_['pred_target_img']
                    data_dict['target_img'] = data_dict_['target_img']
                if i==1:
                    data_dict = data_dict_
            else:

                opt.zero_grad()
                if check_grads:
                    data_dict['source_img'] = data_dict['source_img'].requires_grad_(True)
                    data_dict['source_img'].retain_grad()

                loss, losses_dict_, visuals_, data_dict_ = self.model(data_dict,
                                                                    'train', i,
                                                                    visualize=output_visuals and i == 0,
                                                                    iteration=iteration, rank = self.rank,
                                                                    epoch = epoch)
                losses_dict.update(losses_dict_)

                if i == 0 and visuals_ is not None:
                    visuals.data = visuals_.data
                data_dict.update(data_dict_)



            if self.args.use_amp:
                with amp.scale_loss(loss, opt, loss_id=i) as scaled_loss:
                    scaled_loss.backward()
            else:
                if self.args.use_torch_ddp:
                    # try:
                    loss.backward()
                    # except Exception as e:
                    #     print(f'Opt index is {i}')

                else:
                    loss.backward()
                    

            opt.step()

        if len(self.shds):
            for shd, max_iter in zip(self.shds, self.shd_max_iters):
                if shd.last_epoch < max_iter:
                    shd.step()

        if not len(visuals):
            visuals = None

        return losses_dict, visuals

def main(args):
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--project_dir', default='/fsx/nikitadrobyshev/EmoPortraits', type=str)
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--dataset_name', default='voxceleb2hq_pairs', type=str)
    parser.add_argument('--dataset_name_test', default='voxceleb2hq_pairs', type=str)
    parser.add_argument('--model_type', default='volumetric_avatar', type=str)

    parser.add_argument('--project_name', default="main_gig", type=str)
    parser.add_argument('--entity', default="animator", type=str)
    
    parser.add_argument('--model_name', default='va', type=str)
    parser.add_argument('--model_checkpoint', default=None, type=str)
    parser.add_argument('--trainer_checkpoint', default=None, type=str)
    parser.add_argument('--log_wandb', default='True', type=args_utils.str2bool, choices=[True, False])

    
    parser.add_argument('--use_sec_dataset', default='False', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--sec_dataset_every', default=2, type=int)
    parser.add_argument('--mead_as_second_every', default=0, type=int)
    
    parser.add_argument('--dataset_name_sec', default='extrime_faces_pairs', type=str)
    parser.add_argument('--dataset_name_test_sec', default='extrime_faces_pairs', type=str)

    parser.add_argument('--dataset_name_mead', default='mead_faces_pairs', type=str)
    parser.add_argument('--dataset_name_test_mead', default='mead_faces_pairs', type=str)


    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--local-rank', type=int)
  
    
    
    parser.add_argument('--use_torch_ddp', default='False', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--print_model', default='True', type=args_utils.str2bool, choices=[True, False])


    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--checkpoint_freq', default=10, type=int)
    parser.add_argument('--latest_checkpoint_freq', default=1, type=int, help='frequency of latest checkpoints creation (in epochs)')
    parser.add_argument('--test_freq', default=1, type=int, help='frequency of testing (in epochs')
    parser.add_argument('--output_visuals', default='True', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--logging_freq', default=50, type=int, help='frequency of train logging (in iterations)')
    parser.add_argument('--visuals_freq', default=500, type=int, help='frequency of train visualization (in iterations)')

    parser.add_argument('--use_amp', default='False', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--custom_test', default='False', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--print_model', default='False', type=args_utils.str2bool, choices=[True, False])
    parser.add_argument('--normalize_losses', default='False', type=args_utils.str2bool, choices=[True, False])
    
    parser.add_argument('--use_amp_autocast', action='store_true')

    parser.add_argument('--amp_opt_level', default='O0', type=str)
    parser.add_argument('--check_grads_of_every_loss', default='False', type=args_utils.str2bool, choices=[True, False])

    args, _ = parser.parse_known_args()

    parser = importlib.import_module(f'datasets.{args.dataset_name}').DataModule.add_argparse_args(parser)

    parser = importlib.import_module(f'models.stage_1.{args.model_type}.{args.model_name}_arguments').VolumetricAvatarConfig.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)