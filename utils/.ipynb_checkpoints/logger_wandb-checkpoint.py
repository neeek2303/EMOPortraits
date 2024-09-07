import torch
import os
import pickle as pkl
import tensorboardX
from torchvision import transforms
import copy
import wandb
from glob import glob
from torch.nn.modules.module import _addindent
from tqdm import tqdm

import numpy as np

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


class Logger(object):
    def __init__(self, args, experiment_dir, rank, model, project_name = "main_model_MP2"):
        super(Logger, self).__init__()



        self.rank = rank


        if self.rank == 0:
            wandb.init(project="main_model_MP2", entity="neeek2303", save_code=True)

            #wandb.run.log_code("/fsx/nikitadrobyshev/sound-avatars/sound/sound_avatars/baseline", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
            #wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
            #code = wandb.Artifact('project-source', type='code')

            co = glob('/fsx/nikitadrobyshev/latent-texture-avatar/*.py')
            co += glob('/fsx/nikitadrobyshev/latent-texture-avatar/datasets/*.py')
            co += glob('/fsx/nikitadrobyshev/latent-texture-avatar/*.txt')
            co += glob('/fsx/nikitadrobyshev/latent-texture-avatar/models/*.py')
            co += glob('/fsx/nikitadrobyshev/latent-texture-avatar/networks/volumetric_avatar/*.py')

            for path in co:
                wandb.save(path)

                #code.add_file(path)
            #wandb.run.use_artifact(code)

            wandb.run.name = str(experiment_dir).split('/')[-1] + '_' + wandb.run.id[:4]
            wandb.run.save()
            wandb.config.update(args) 




        self.ddp = args.num_gpus > 1
        self.logging_freq = args.logging_freq
        self.visuals_freq = args.visuals_freq
        self.batch_size = args.batch_size
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.rank = rank

        self.train_iter = 0
        self.epoch = 0
        self.output_train_logs = not (self.train_iter + 1) % self.logging_freq
        self.output_train_visuals = self.visuals_freq > 0 and not (self.train_iter + 1) % self.visuals_freq

        self.to_image = transforms.ToPILImage()
        self.losses_buffer = {'train': {}, 'test': {}}

        if self.rank == 0:
            for phase in ['train', 'test']:
                os.makedirs(self.experiment_dir / 'images' / phase, exist_ok=True)

            self.losses = {'train': {}, 'test': {}}
            # self.writer = tensorboardX.SummaryWriter(self.experiment_dir)

            s = ''
            for i in vars(args).items():
                s += str(i) + '    \n'
            wandb.log({'params': s}, step=0)
            ms = torch_summarize(model)
            wandb.log({'model': ms}, step=0)

            wandb.watch(model)



    def log(self, phase, losses_dict = None, visuals = None, epoch_end = False):
        if losses_dict is not None:
            for name, loss in losses_dict.items():
                if name in self.losses_buffer[phase].keys():
                    if type(loss) == torch.Tensor:
                        self.losses_buffer[phase][name].append(loss.detach())
                    else:
                        print(f'loss {name} has wrong type {type(loss)} with value {loss}')
                else:
                    self.losses_buffer[phase][name] = [loss.detach()]

        if phase == 'train':
            self.train_iter += 1

            if self.output_train_logs:
                self.output_logs(phase)

            if self.output_train_visuals and visuals is not None:
                self.output_visuals(phase, visuals)
            self.output_train_logs = not (self.train_iter + 1) % self.logging_freq
            self.output_train_visuals = self.visuals_freq > 0 and not (self.train_iter + 1) % self.visuals_freq

        elif phase == 'test' and epoch_end:
            self.epoch += 1
            self.output_logs(phase)

            if visuals is not None:
                self.output_visuals(phase, visuals)

    def output_logs(self, phase):
        # Average the buffers and flush
        names = list(self.losses_buffer[phase].keys())
        losses = []
        for losses_ in self.losses_buffer[phase].values():
            # losses_ = [torch.Tensor(i.cpu()) for i in losses_]
            losses.append(torch.stack(losses_).mean())
        losses = torch.stack(losses)

        self.losses_buffer[phase] = {}

        if self.ddp:
            # Synchronize buffers across GPUs
            losses_ = torch.zeros(size=(torch.distributed.get_world_size(), len(losses)), dtype=losses.dtype, device=losses.device)
            losses_[self.rank] = losses
            torch.distributed.reduce(losses_, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                losses = losses_.mean(0)

        file = self.experiment_dir / 'file.txt'
        open(file, 'w').close()

        if self.rank == 0:
            for name, loss in zip(names, losses):
                loss = loss.item()
                if name in self.losses[phase].keys():
                    self.losses[phase][name].append(loss)
                else:
                    self.losses[phase][name] = [loss]

                # self.writer.add_scalar(name, loss, self.train_iter)
                wandb.log({name:loss}, step=self.train_iter)

            tqdm.write(f'Iter {self.train_iter:06d} ' + ', '.join(f'{name}: {losses[-1]:.3f}' for name, losses in self.losses[phase].items()))

    def output_visuals(self, phase, visuals):
        device = str(visuals.device)

        if self.ddp and device != 'cpu':
            # Synchronize visuals across GPUs
            c, h, w = visuals.shape[1:]            
            b = self.batch_size if phase == 'train' else 1
            visuals_ = torch.zeros(size=(torch.distributed.get_world_size(), b, c, h, w), dtype=visuals.dtype, device=visuals.device)
            visuals_[self.rank, :visuals.shape[0]] = visuals
            torch.distributed.reduce(visuals_, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                visuals = visuals_.view(-1, c, h, w)

        if device != 'cpu':
            # All visuals are reduced, save only one image
            name = f'{self.train_iter:06d}.jpg'
        else:
            # Save all images
            name = f'{self.train_iter:06d}_{self.rank}.jpg'

        # if self.rank == 0 or device == 'cpu':
        if self.rank == 0 or device == 'cpu':
            visuals = torch.cat(visuals.split(1, 0), 2)[0] # cat batch dim in lines w.r.t. height
            visuals = visuals.cpu()

            # Save visuals
            image = self.to_image(visuals)
            image.save(self.experiment_dir / 'images' / phase / name)

            if self.rank == 0:
                # self.writer.add_image(f'{phase}_images', visuals, self.train_iter)
                wandb.log({f'{phase}_images_{self.rank}': [wandb.Image(image)]}, step=self.train_iter)

    def state_dict(self):
        state_dict = {
            'losses': self.losses,
            'train_iter': self.train_iter,
            'epoch': self.epoch}

        return state_dict

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']
        self.train_iter = state_dict['train_iter']
        self.epoch = state_dict['epoch']