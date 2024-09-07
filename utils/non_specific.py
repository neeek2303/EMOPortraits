import torch
from torch import nn
import torch.nn.functional as F
from . import spectral_norm, weight_init, point_transforms
import random
import numpy as np
from scipy import linalg
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap
from ibug.roi_tanh_warping import roi_tanh_polar_restore, roi_tanh_polar_warp

from torchvision import transforms
to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn


from .utils_keypoints import procrustes, get_transform_matrix, transform_landmarks, get_scale_matrix

def align_keypoints(source_kp, pred_kp, nose=False):
    if nose:
        l = [27, 28, 29, 30]
    else:
        l = [30,  36, 39, 42,  45]
    out  = procrustes(torch.tensor(pred_kp[l]), torch.tensor(source_kp[l]))
    
    if nose:
        matrix = get_scale_matrix(out[1], out[2], out[0])
    else:
        matrix = get_transform_matrix(out[1], out[2], out[0])

    new_pred = transform_landmarks(torch.tensor(pred_kp), matrix)
    return new_pred, matrix

def align_keypoints_torch(source_kp, pred_kp, nose=False):
    out_np_b = []
    matrix_b = []
    for s_kp, p_kp in zip(source_kp, pred_kp):
        out_np, m = align_keypoints(s_kp.detach().cpu().numpy(), p_kp.detach().cpu().numpy(), nose=nose)
        out_np_b.append(out_np)
        matrix_b.append(m)

    return torch.stack(out_np_b, dim=0), torch.cat(matrix_b, dim=0)


def calculate_obj_params(obj, opt_net_names=None, opt_tensor_names=None, opt_dis_names=None, prt=False, net_sf = '_nw',  tensor_sf = '_ts', dis_sf = '_ds'):

    obj.opt_net_names = opt_net_names if opt_net_names!= None else [i for i in dir(obj) if net_sf in i]

    obj.opt_tensor_names = opt_tensor_names if opt_tensor_names!= None else [i for i in dir(obj) if tensor_sf in i]

    obj.opt_dis_names = opt_dis_names if opt_dis_names!= None else [i for i in dir(obj) if dis_sf in i]
    
    if prt:
        print("Net names: {obj.opt_net_names} \n")
        print("Tensor names: {obj.opt_tensor_names} \n")

    obj.net_param_dict = {} 
    obj.total_params = 0
    for net_name in obj.opt_net_names:
        try:
            net = getattr(obj, net_name)
            net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            obj.net_param_dict[net_name] = net_params
            obj.total_params+=net_params
        except Exception as e:
            pass

    for net_name in obj.opt_dis_names:
        try:
            net = getattr(obj, net_name)
            net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            obj.net_param_dict[net_name] = net_params
            obj.total_params+=net_params
        except Exception as e:
            pass

    for tensor_name in obj.opt_tensor_names:
        try:
            tensor = getattr(obj, tensor_name)
            tensor_params = len(tensor.flatten())
            obj.net_param_dict[tensor_name] = tensor_params
            obj.total_params+=tensor_params
        except Exception as e:
            print(e)




def pca_metrics(v_list, n_comp=512):
    all_vectors = torch.cat(v_list, dim=0).numpy()
    vec_df = pd.DataFrame(all_vectors)
    scaler = StandardScaler()
    _ = scaler.fit_transform(all_vectors)
    X = scaler.transform(vec_df)
    dfx = pd.DataFrame(data=X)
    pca = PCA(n_components=n_comp)
    dfx_pca = pca.fit(dfx)
    a = torch.tensor(np.sum((dfx_pca.explained_variance_ratio_*100)>1)).float().cuda()
    b = torch.tensor(np.sum((dfx_pca.explained_variance_ratio_*100)>0.1)).float().cuda()
    c = torch.tensor(np.sum((dfx_pca.explained_variance_ratio_*100)>0.01)).float().cuda()
    auc = torch.tensor(sklearn.metrics.auc([i+1 for i in range(n_comp)], dfx_pca.explained_variance_ratio_[:n_comp])).float().cuda()
    cum_explain = np.cumsum(dfx_pca.explained_variance_ratio_)
    n_99 =  torch.tensor(list((cum_explain>0.99)).index(True)).float().cuda()
    n_999 =  torch.tensor(list((cum_explain>0.999)).index(True)).float().cuda()
    n_9999 =  torch.tensor(list((cum_explain>0.9999)).index(True)).float().cuda()

    return dfx_pca.explained_variance_ratio_, auc, n_99, n_999, n_9999





@torch.no_grad()
def get_face_warp(obj, grid, params_ffhq):
    grid = grid.view(grid.shape[0], -1, 2)
    face_warp = point_transforms.align_ffhq_with_zoom(grid, params_ffhq)
    face_warp = face_warp.view(face_warp.shape[0], obj.args.aug_warp_size, obj.args.aug_warp_size, 2)

    return face_warp


def get_mixing_theta(args, source_theta, target_theta, thetas_pool=[], random_theta = True):
        source_theta = source_theta[:, :3, :]
        target_theta = target_theta[:, :3, :]
        N = 1
        B = source_theta.shape[0] // N
        T = target_theta.shape[0] // B

        source_theta_ = np.stack([np.eye(4) for i in range(B)])
        target_theta_ = np.stack([np.eye(4) for i in range(B * T)])

        source_theta = source_theta.view(B, N, *source_theta.shape[1:])[:, 0]  # take theta from the first source image

        if random_theta:
            r = random.randint(0, 1)
            if B==2:
                thetas_pool.append(target_theta)
                if len(thetas_pool)>=50:
                    thetas_pool.pop(0)
                r = random.randint(0, len(thetas_pool)-1)
                th = thetas_pool[r]
                p = random.randint(0, 1)
                target_theta = target_theta if p<=0 else th
                target_theta = target_theta.view(B, T, 3, 4).roll(0, dims=0).view(B * T, 3, 4)  # shuffle target poses
            else:
                target_theta = target_theta.view(B, T, 3, 4).roll(0, dims=0).view(B * T, 3, 4)  # shuffle target poses
        else:
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
                        if args.old_mix_pose:
                            pred_mixing_theta.append(target_translation[b * T + t] @ target_rotation @ source_stretch)
                        else:
                            pred_mixing_theta.append(source_stretch * target_stretch.mean() / source_stretch.mean() @ target_rotation @ target_translation[b * T + t])

        pred_mixing_theta = np.stack(pred_mixing_theta)

        return torch.from_numpy(pred_mixing_theta)[:, :3].type(source_theta.type()).to(source_theta.device), thetas_pool

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def keypoints_to_heatmaps(keypoints, img):
    HEATMAPS_VAR = 1e-2
    s = img.shape[2]

    keypoints = keypoints[..., :2]  # use 2D projection of keypoints

    return kp2gaussian(keypoints, img.shape[2:], HEATMAPS_VAR)




class FaceParsingBUG(object):
    def __init__(self, device = 'cuda'):
        super(FaceParsingBUG, self).__init__()
        decoder = 'fcn'
        encoder = 'rtnet50'
        num_classes = 14
        weights = None
        threshold = 0.8

        self.face_parser = RTNetPredictor(
            device=device, ckpt=weights, encoder=encoder, decoder=decoder, num_classes=num_classes)

        colormap = label_colormap(num_classes)
        self.face_detector = RetinaFacePredictor(threshold=threshold, device=device,
                                            model=(RetinaFacePredictor.get_model('mobilenet0.25')))

    def get_lips(self, img, faces=None):
        h, w = img.shape[-2:]
        img_cv2 =  img.cpu().detach().numpy().transpose(1, 2, 0)*255
        _faces = self.face_detector(img_cv2, rgb=True)
        faces = faces if faces is not None else _faces

        
        if faces.shape==(0,15):
            masks = None
            logits = None
            faces = None
            logits_s = None
        else:
            img = img.unsqueeze(0)
            masks, logits, bboxes_tensor, img = self.face_parser.predict_img(img, to_tensor(faces), rgb=True)
            logits_s = torch.softmax(logits, 1)
            logits_s[:, 0] = 1 - logits_s[:, 0]
            logits_s = roi_tanh_polar_restore(
                logits_s, bboxes_tensor, h, w, keep_aspect_ratio=True
            )
        return masks, logits, logits_s, faces



