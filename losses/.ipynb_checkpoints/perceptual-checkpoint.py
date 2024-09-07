import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms
from typing import Union

from utils import misc
from .emotion_models import VGG
from .emotion_models import ResNet18_ARM___RAF
from .emotion_models import MobileFaceNet
from .senet50_ft_dag import senet50_ft_dag

class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.
    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the inputsut images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
        use_fp16 (bool) : If ``True``, use cast networks and inputs to FP16
    """

    def __init__(self, 
                 network='vgg19', 
                 layers=('relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'), 
                 # weights=(0.03125, 0.0625, 0.125, 0.25, 1.0),
                 weights=(0.2, 0.2, 0.2, 0.2, 0.2),
                 criterion='l1',
                 resize=False, 
                 resize_mode='bilinear',
                 instance_normalized=False,
                 replace_maxpool_with_avgpool=False,
                 num_scales=1,
                 use_fp16=False,
                 use_conf=True,
                 resize_size = 224,
                 gray=False,
                 scale_factor = 0.5,
                 apply_normalization = True,
                 face_norm = False
                ) -> None:
        super(PerceptualLoss, self).__init__()
        if isinstance(layers, str):
            layers = [layers]
        self.use_conf = use_conf
        self.gray = gray
        self.face_norm =face_norm
        self.network = network
        self.apply_normalization = apply_normalization
        self.scale_factor = scale_factor
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]
        self.resize_size = resize_size
        self.return_landmarks = False
        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        elif network == 'vgg16':
            self.model = _vgg16(layers)
        elif network == 'alexnet':
            self.model = _alexnet(layers)
        elif network == 'inception_v3':
            self.model = _inception_v3(layers)
        elif network == 'resnet50':
            self.model = _resnet50(layers)
        elif network == 'robust_resnet50':
            self.model = _robust_resnet50(layers)
        elif network == 'vgg_face_dag':
            self.model = _vgg_face_dag(layers)
        elif network == 'face_parsing':
            self.model = _bisenet_FP(layers)
        elif network == 'face_resnet':
            self.model = _face_resnet(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        if replace_maxpool_with_avgpool:
	        for k, v in self.model.network._modules.items():
	        	if isinstance(v, nn.MaxPool2d):
	        		self.model.network._modules[k] = nn.AvgPool2d(2)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSEloss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized
        self.fp16 = use_fp16
        if self.fp16:
            self.model.half()

    @torch.cuda.amp.autocast(True)
    def forward(self, 
                inputs: Union[torch.Tensor, list], 
                target: torch.Tensor,
                confs_ms: torch.Tensor,
                mask = None,
                num_scales=None) -> Union[torch.Tensor, list]:
        r"""Perceptual loss forward.
        Args:
           inputs (4D tensor or list of 4D tensors) : inputsut tensor.
           target (4D tensor) : Ground truth tensor, same shape as the inputsut.
        Returns:
           (scalar tensor or list of tensors) : The perceptual loss.
        """

        if num_scales is None:
            num_scales = self.num_scales


        if isinstance(inputs, list):
            # Concat alongside the batch axis
            input_is_a_list = True
            num_chunks = len(inputs)
            inputs = torch.cat(inputs)
        else:
            input_is_a_list = False
        # Perceptual loss should operate in eval mode by default.
        if mask:
            inputs = inputs * mask + inputs.detach() * (1 - mask)


        self.model.eval()
        if self.apply_normalization:
            inputs, target = \
                misc.apply_imagenet_normalization(inputs), \
                misc.apply_imagenet_normalization(target)

        if self.resize:
            inputs = F.interpolate(
                inputs, mode=self.resize_mode, size=(self.resize_size , self.resize_size ),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(self.resize_size , self.resize_size ),
                align_corners=False)

            if self.gray:

                for i, j in enumerate([0.299, 0.587, 0.114]):
                    inputs[:, i, :, :].unsqueeze(1) * j
                one_channel_input = inputs[:, 0, :, :].unsqueeze(1) * 0.299 + inputs[:, 1, :, :].unsqueeze(1) * 0.587 + inputs[:, 2, :, :].unsqueeze(1) * 0.114
                one_channel_target = target[:, 0, :, :].unsqueeze(1) * 0.299 + target[:, 1, :, :].unsqueeze(1) * 0.587 + target[:, 2, :, :].unsqueeze(1) * 0.114
                inputs = torch.cat([one_channel_input, one_channel_input, one_channel_input], dim=1)
                target = torch.cat([one_channel_target, one_channel_target, one_channel_target], dim=1)

            if self.face_norm:
                mean = inputs.new_tensor((131.0912, 103.8827, 91.4953)).view(1, 3, 1, 1)
                inputs = inputs - mean
                target = target - mean
        # Evaluate perceptual loss at each scale.

        loss = 0
        penalty = 0
        for scale in range(num_scales):
            if self.fp16:
                input_features = self.model(inputs.half())
                with torch.no_grad():
                    target_features = self.model(target.half())
            else:
                input_features = self.model(inputs)
                with torch.no_grad():
                    target_features = self.model(target)

            for k, (layer, weight) in enumerate(zip(self.layers, self.weights)):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]

                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if input_is_a_list:
                    target_feature = torch.cat([target_feature] * num_chunks)
                dist = (input_feature - target_feature).abs()
                if confs_ms is not None :
                    dist = dist * confs_ms[k]
                    penalty_k = -confs_ms[k].log()
                    penalty += penalty_k.mean() * weight

                loss += weight * dist.mean()

                # print(input_feature.shape, target_feature.shape, dist.shape, confs_ms[k].shape, torch.min(confs_ms[k]), torch.mean(confs_ms[k]), torch.max(confs_ms[k]), weight )
            # Downsample the inputsut and target.

            if scale != num_scales - 1:
                inputs = F.interpolate(
                    inputs, mode=self.resize_mode, scale_factor=self.scale_factor,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=self.scale_factor,
                    align_corners=False, recompute_scale_factor=True)
                if confs_ms is not None:
                    confs_ms = [F.avg_pool2d(confs, stride=2, kernel_size=2) for confs in confs_ms]

        loss /= num_scales
        if confs_ms is not None:
            penalty /= num_scales
            return loss, penalty
        elif self.return_landmarks:
            return loss, (input_features[self.layers[-1]], target_features[self.layers[-1]])
        else:
            return loss, None

    def train(self, mode: bool = True):
        return self


class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.
    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers

        for m in self.network.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):

            layer_name = self.layer_name_mapping.get(i, None)

            x = layer(x)


            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output



class _PerceptualNetworkFP(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.
    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])


    def forward(self, x):
        r"""Extract perceptual features."""

        h, w = x.shape[2:]
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        x = self.network.conv1(x)
        x = F.relu(self.network.bn1(x))
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        feat8 = self.network.layer2(x) # 1/8
        feat16 = self.network.layer3(feat8) # 1/16
        feat32 = self.network.layer4(feat16) # 1/32

        output = {'feat4':x,'feat8':feat8, 'feat16':feat16, 'feat32':feat32 }

        return output
    



class _PerceptualNetwork_face(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.
    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layers):
        super().__init__()
        self.network = network
        self.layers = layers

        for m in self.network.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}

        out = self.network(x)

        for i, l in enumerate(out):
            output[self.layers[i]] = l
        return output

def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg16(layers):
    r"""Get vgg16 layers"""
    network = torchvision.models.vgg16(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          18: 'relu_4_1',
                          20: 'relu_4_2',
                          22: 'relu_4_3',
                          25: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _alexnet(layers):
    r"""Get alexnet layers"""
    network = torchvision.models.alexnet(pretrained=True).features
    layer_name_mapping = {0: 'conv_1',
                          1: 'relu_1',
                          3: 'conv_2',
                          4: 'relu_2',
                          6: 'conv_3',
                          7: 'relu_3',
                          8: 'conv_4',
                          9: 'relu_4',
                          10: 'conv_5',
                          11: 'relu_5'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _inception_v3(layers):
    r"""Get inception v3 layers"""
    inception = torchvision.models.inception_v3(pretrained=True)
    network = nn.Sequential(inception.Conv2d_1a_3x3,
                            inception.Conv2d_2a_3x3,
                            inception.Conv2d_2b_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Conv2d_3b_1x1,
                            inception.Conv2d_4a_3x3,
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            inception.Mixed_5b,
                            inception.Mixed_5c,
                            inception.Mixed_5d,
                            inception.Mixed_6a,
                            inception.Mixed_6b,
                            inception.Mixed_6c,
                            inception.Mixed_6d,
                            inception.Mixed_6e,
                            inception.Mixed_7a,
                            inception.Mixed_7b,
                            inception.Mixed_7c,
                            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    layer_name_mapping = {3: 'pool_1',
                          6: 'pool_2',
                          14: 'mixed_6e',
                          18: 'pool_3'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _resnet50(layers):
    r"""Get resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=True)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _robust_resnet50(layers):
    r"""Get robust resnet50 layers"""
    resnet50 = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.utils.model_zoo.load_url(
        'http://andrewilyas.com/ImageNet.pt')
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith('module.model.'):
            new_state_dict[k[13:]] = v
    resnet50.load_state_dict(new_state_dict)
    network = nn.Sequential(resnet50.conv1,
                            resnet50.bn1,
                            resnet50.relu,
                            resnet50.maxpool,
                            resnet50.layer1,
                            resnet50.layer2,
                            resnet50.layer3,
                            resnet50.layer4,
                            resnet50.avgpool)
    layer_name_mapping = {4: 'layer_1',
                          5: 'layer_2',
                          6: 'layer_3',
                          7: 'layer_4'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)


def _vgg_face_dag(layers):
    r"""Get vgg face layers"""
    network = torchvision.models.vgg16(num_classes=2622).features
    state_dict = torch.utils.model_zoo.load_url(
        'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/'
        'vgg_face_dag.pth')
    layer_name_mapping = {
        0: 'conv1_1',
        2: 'conv1_2',
        5: 'conv2_1',
        7: 'conv2_2',
        10: 'conv3_1',
        12: 'conv3_2',
        14: 'conv3_3',
        17: 'conv4_1',
        19: 'conv4_2',
        21: 'conv4_3',
        24: 'conv5_1',
        26: 'conv5_2',
        28: 'conv5_3'}
    new_state_dict = {}
    for k, v in layer_name_mapping.items():
        new_state_dict[str(k) + '.weight'] =\
            state_dict[v + '.weight']
        new_state_dict[str(k) + '.bias'] = \
            state_dict[v + '.bias']
    network.load_state_dict(new_state_dict)
    return _PerceptualNetwork(network, layer_name_mapping, layers)

from repos.face_par_off.model import BiSeNet
import os
def _bisenet_FP(layers):
    r"""Get vgg face layers"""
    n_classes = 19
    network = BiSeNet(n_classes=n_classes)
    project_dir = '/fsx/nikitadrobyshev/EmoPortraits'
    path_to_face_parsing = f'{project_dir}/repos/face_par_off'
    state_dict_p = os.path.join(f'{path_to_face_parsing}/res/cp/79999_iter.pth')
    network.load_state_dict(torch.load(state_dict_p, map_location='cpu'))
    network.eval()
    network = network.cp.resnet

    layer_name_mapping = {
        0: 'conv1',
        1: 'bn1',
        2: 'maxpool',
        3: 'layer1',
        4: 'layer2',
        5: 'layer3',
        6: 'layer4',
        }

    
    return _PerceptualNetworkFP(network, layer_name_mapping, layers)




def _face_resnet(layers):
    r"""Get vgg face layers"""
    network = senet50_ft_dag(
        weights_path='/fsx/nikitadrobyshev/EmoPortraits/losses/loss_model_weights/senet50_ft_dag.pth')
    network.eval()

    # c = list(network.children())
    # network = nn.Sequential(*c)

    layer_name_mapping = {50: 'relu1',
                          63: 'relu2',
                          74: 'relu3',
                          97: 'relu4',
                          111: 'relu5',
                          121: 'relu6',
                          132: 'relu7',
                          141: 'relu8',
                          152: 'relu9',
                          169: 'relu10',
                          183: 'relu11',
                          197: 'relu12',
                          213: 'relu13',
                          227: 'relu14',
                          235: 'relu15',
                          }

    return _PerceptualNetwork_face(network, layers)