import torch
from torch import nn
import torch.nn.functional as F
from utils.prepare_for_landmark import prepare_face_for_landmarks
import numpy as np
import losses
from torchvision import transforms
import sys
sys.path.append('/fsx/nikitadrobyshev/EmoPortraits')
from repos.MODNet.src.models.modnet import MODNet
import cv2
from utils import misc

def calc_train_losses(obj, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0, iteration=0):
    losses_dict = {}

    if mode == 'dis':
        if obj.args.adversarial_weight > 0:
            losses_dict['dis_adversarial'] = (
                    obj.weights['adversarial'] *
                    obj.adversarial_loss(
                        real_scores=data_dict['real_score_dis'],
                        fake_scores=data_dict['fake_score_dis'],
                        mode='dis'))

        if obj.args.use_second_dis:
            losses_dict['dis_adversarial_2'] = (
                    obj.adversarial_loss(
                        real_scores=data_dict['real_score_dis_mix'],
                        fake_scores=data_dict['fake_score_dis_mix'],
                        mode='dis'))

    if mode == 'gen':
        if obj.args.adversarial_weight > 0 and obj.pred:
            losses_dict['gen_adversarial'] = (
                    obj.weights['adversarial'] *
                    obj.weights['adversarial_gen']*
                    obj.adversarial_loss(
                        fake_scores=data_dict['fake_score_gen'],
                        mode='gen'))

            losses_dict['feature_matching'] = (
                    obj.weights['feature_matching'] *
                    obj.feature_matching_loss(
                        real_features=data_dict['real_feats_gen'],
                        fake_features=data_dict['fake_feats_gen']))

        if obj.args.use_second_dis and obj.mix:
            losses_dict['gen_adversarial_2'] = (
                                                obj.weights['adversarial_gen_2'] *
                                                obj.adversarial_loss(
                                                    fake_scores=data_dict['fake_score_gen_mix'],
                                                    mode='gen'))
        s = obj.args.resize_s2
        resize = lambda img: F.interpolate(img, mode='area', size=(s, s))
        resize2 = lambda img: F.interpolate(img, mode='area', size=(s//2, s//2))

        # print(obj.weights['vgg19'])
        # print(obj.vgg19_loss(data_dict['pred_target_img_ffhq'], data_dict['target_img_ffhq'], None))



        if obj.pred:
            data_dict['pred_target_img_ffhq_RESIZED'] = resize(data_dict['pred_target_img_ffhq'])
            data_dict['pred_target_img_RESIZED'] = resize(data_dict['pred_target_img'])
            # data_dict['pred_target_img_ffhq_RESIZED'] = resize(data_dict['pred_target_img_ffhq'])
            # data_dict['pred_target_img_RESIZED'] = resize(data_dict['pred_target_img'])

            if obj.weights['vgg19']:
                losses_dict['vgg19'], _ = obj.vgg19_loss(data_dict['pred_target_img_ffhq'], data_dict['target_img_ffhq'], None)
                losses_dict['vgg19'] *= obj.weights['vgg19']


            if obj.weights['l1_weight']:
                losses_dict['L1_ffhq'] = obj.weights['l1_weight'] * obj.l1_loss(data_dict['pred_target_img_ffhq'], data_dict['target_img_ffhq'])
                losses_dict['L1_ffhq_diff'] = (obj.args.diff_ratio - 1)*obj.weights['l1_weight'] * obj.l1_loss(data_dict['pred_target_img_ffhq']*data_dict['target_add_ffhq_pred_mask'], data_dict['target_img_ffhq']*data_dict['target_add_ffhq_pred_mask'])



            if obj.weights['vgg19']:
                losses_dict['vgg19_cycle'], _ = obj.vgg19_loss(data_dict['pred_target_img_ffhq_RESIZED'],data_dict['pred_target_img_RESIZED'], None)
                losses_dict['vgg19_cycle'] *= obj.weights['vgg19'] * obj.args.cycle_stage2

            # if obj.weights['l1_weight']:
            #     losses_dict['L1_ffhq_cycle'] = obj.weights['l1_weight'] * obj.args.cycle_stage2 * obj.l1_loss(data_dict['pred_target_img_ffhq_RESIZED'], data_dict['pred_target_img_RESIZED'])

            data_dict['pred_target_img_ffhq_RESIZED'] = resize2(data_dict['pred_target_img_ffhq'])
            data_dict['pred_target_img_RESIZED'] = resize2(data_dict['pred_target_img'])

            if obj.weights['l1_weight']:
                losses_dict['L1_ffhq_cycle'] = obj.weights['l1_weight'] * obj.args.cycle_stage2 * obj.l1_loss(data_dict['pred_target_img_ffhq_RESIZED'], data_dict['pred_target_img_RESIZED'])


        data_dict['pred_mixing_img_ffhq_RESIZED'] = resize(data_dict['pred_mixing_img_ffhq'])
        data_dict['resized_pred_mixing_img_RESIZED'] = resize(data_dict['pred_mixing_img'])
        if obj.mix:
            if obj.weights['vgg19']:
                losses_dict['vgg19_mix'], _ = obj.vgg19_loss(data_dict['pred_mixing_img_ffhq_RESIZED'],data_dict['resized_pred_mixing_img_RESIZED'], None)
                losses_dict['vgg19_mix'] *= obj.weights['vgg19'] * obj.args.cycle_stage2

            # if obj.weights['l1_weight']:
            #     losses_dict['L1_ffhq_mix'] = obj.weights['l1_weight'] * obj.args.cycle_stage2 * obj.l1_loss(data_dict['pred_mixing_img_ffhq_RESIZED'], data_dict['resized_pred_mixing_img_RESIZED'])

            data_dict['pred_mixing_img_ffhq_RESIZED'] = resize2(data_dict['pred_mixing_img_ffhq'])
            data_dict['resized_pred_mixing_img_RESIZED'] = resize2(data_dict['pred_mixing_img'])

            if obj.weights['l1_weight']:
                losses_dict['L1_ffhq_mix'] = obj.weights['l1_weight'] * obj.args.cycle_stage2 * obj.l1_loss(data_dict['pred_mixing_img_ffhq_RESIZED'], data_dict['resized_pred_mixing_img_RESIZED'])


        #
        # n = data_dict['source_img'].shape[0]
        # t = data_dict['target_img'].shape[0]
        # inputs_orig_face_aligned = F.grid_sample(
        #     torch.cat([data_dict['pred_target_img'], data_dict['target_img']]).float(),
        #     data_dict['align_warp'].float())
        # data_dict['pred_target_img_face_align'], data_dict[
        #     'target_img_align_orig'] = inputs_orig_face_aligned.split([n, t], dim=0)
        #
        # if obj.weights['vgg19_face']:
        #     if ffhq_per_b > 0:
        #         losses_dict['vgg19_face'], _ = obj.vgg19_loss_face(
        #             data_dict['pred_target_img_face_align'][:-ffhq_per_b],
        #             data_dict['target_img_align_orig'][:-ffhq_per_b], None)
        #         losses_dict['vgg19_face'] *= obj.weights['vgg19_face'] * (3/4)
        #
        #         losses_dict['vgg19_face_ffhq'], _ = obj.vgg19_loss_face(
        #             data_dict['pred_target_img_face_align'][-ffhq_per_b:],
        #             data_dict['target_img_align_orig'][-ffhq_per_b:], None)
        #         losses_dict['vgg19_face_ffhq'] *= obj.weights['vgg19_face'] * (1/4)
        #     else:
        #         losses_dict['vgg19_face'], _ = obj.vgg19_loss_face(data_dict['pred_target_img_face_align'],
        #                                                            data_dict['target_img_align_orig'], None)
        #         losses_dict['vgg19_face'] *= obj.weights['vgg19_face']




    loss = 0
    for k, v in losses_dict.items():
        try:
            loss += v
        except Exception as e:
            print(e, ' Loss adding error')
            print(k, v, loss)
            losses_dict[k] = v[0]
            if v.shape[0] > 1:
                raise ValueError
        finally:
            pass

    return loss, losses_dict


def calc_test_losses(obj, data_dict: dict):
    pred_dtype = data_dict['pred_target_img'].type()
    dtype = data_dict['target_img'].type()
    b = data_dict['pred_target_img'].shape[0]
    if pred_dtype != dtype:
        data_dict['pred_target_img'] = data_dict['pred_target_img'].type(dtype)

    # with torch.no_grad():
        # face_vector_target, target_face = obj.get_face_vector.forward(data_dict['target_img'])
        # face_vector_mixing, mixing_face = obj.get_face_vector.forward(data_dict['pred_mixing_img'])
        # face_vector_target_resnet = obj.get_face_vector_resnet.forward(target_face)
        # face_vector_mixing_resnet = obj.get_face_vector_resnet.forward(mixing_face)
        #
        # face_vector_target_resnet_no_crop = obj.get_face_vector_resnet.forward(data_dict['target_img'])
        # face_vector_mixing_resnet_no_crop = obj.get_face_vector_resnet.forward(data_dict['pred_mixing_img'])

    y = torch.tensor([1] * b).to(data_dict['target_img'].device)

    losses_dict = {
        'l1_ffhq': obj.l1_loss(data_dict['pred_target_img'], data_dict['target_img']),
        # 'ssim': obj.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean(),
        # 'psnr': obj.psnr(data_dict['pred_target_img'], data_dict['target_img']),
        # 'lpips': obj.lpips(data_dict['pred_target_img'], data_dict['target_img']),
        # 'face_vgg': obj.l1_loss(face_vector_target, face_vector_mixing),
        # 'face_resnet': obj.l1_loss(face_vector_target_resnet, face_vector_mixing_resnet),
        # 'face_resnet_no_crop': obj.l1_loss(face_vector_target_resnet_no_crop, face_vector_mixing_resnet_no_crop),
        # 'face_vgg_cos': obj.cosin_sim(face_vector_target.view(b, -1), face_vector_mixing.view(b, -1), y),
        # 'face_resnet_cos': obj.cosin_sim(face_vector_target_resnet.view(b, -1), face_vector_mixing_resnet.view(b, -1),
        #                                  y),
        # 'face_resnet_no_crop_cos': obj.cosin_sim(face_vector_target_resnet_no_crop.view(b, -1),
        #                                          face_vector_mixing_resnet_no_crop.view(b, -1), y)
    }

    if obj.sep_test_losses:
        losses_dict['ssim person'] = obj.ssim(data_dict['pred_target_img'] * data_dict['target_mask'],
                                              data_dict['target_img'] * data_dict['target_mask']).mean()
        losses_dict['psnr person'] = obj.psnr(data_dict['pred_target_img'] * data_dict['target_mask'],
                                              data_dict['target_img'] * data_dict['target_mask'])
        losses_dict['lpips person'] = obj.lpips(data_dict['pred_target_img'] * data_dict['target_mask'],
                                                data_dict['target_img'] * data_dict['target_mask'])
        # losses_dict['ssim back'] = obj.ssim(data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
        #                                     data_dict['target_img'] * (1 - data_dict['target_mask'])).mean()
        # losses_dict['psnr back'] = obj.psnr(data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
        #                                     data_dict['target_img'] * (1 - data_dict['target_mask']))
        # losses_dict['lpips back'] = obj.lpips(data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
        #                                       data_dict['target_img'] * (1 - data_dict['target_mask']))
    #
    # if obj.args.image_size > 160:
    #     losses_dict['ms_ssim'] = obj.ms_ssim(data_dict['pred_target_img'], data_dict['target_img']).mean()
    #     if obj.sep_test_losses:
    #         losses_dict['ms_ssim person'] = obj.ms_ssim(data_dict['pred_target_img'] * data_dict['target_mask'],
    #                                                     data_dict['target_img'] * data_dict['target_mask']).mean()
    #         losses_dict['ms_ssim back'] = obj.ms_ssim(
    #             data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
    #             data_dict['target_img'] * (1 - data_dict['target_mask'])).mean()
    return losses_dict


def init_losses(obj, args):
    # if obj.weights['adversarial']:
    obj.adversarial_loss = losses.AdversarialLoss()

    # if obj.weights['feature_matching']:
    obj.feature_matching_loss = losses.FeatureMatchingLoss()

    if obj.weights['gaze']:
        obj.gaze_loss = losses.GazeLoss(device='cuda', gaze_model_types=['vgg16']) #weights=(0.0, 0.0, 0.0, 0.0, 1.0) , weights=(0.3, 0.25, 0.2, 0.15, 0.1)

    if obj.weights['vgg19']:
        obj.vgg19_loss = losses.PerceptualLoss(num_scales=2, use_fp16=False)

    if obj.weights['vgg19_face']:
        obj.vgg19_loss_face = losses.PerceptualLoss(num_scales=2, network='vgg_face_dag',
                                                    layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                                    resize=True, weights=(0.03125, 0.0625, 0.125, 0.25, 1.0), use_fp16=False)



    if obj.weights['warping_reg']:
        obj.warping_reg_loss = losses.WarpingRegLoss()

    obj.l1_loss = nn.L1Loss()
    obj.cosin_sim = torch.nn.CosineEmbeddingLoss(margin=0.3)
    obj.cosin_sim_2 = torch.nn.CosineEmbeddingLoss(margin=0.5, reduce=False)
    obj.cosin_dis = torch.nn.CosineSimilarity()
    obj.affine_match_loss = losses.AffineLoss(args)
    obj.warp_reg_loss = losses.WarpReg(args)
    obj.ssim = losses.SSIM(data_range=1, size_average=True, channel=3)
    obj.ms_ssim = losses.MS_SSIM(data_range=1, size_average=True, channel=3)
    obj.psnr = losses.PSNR()
    obj.lpips = losses.LPIPS()
    obj.expansion_factor = 1
    obj.sep_test_losses = args.sep_test_losses




def prepare_input_data(obj, data_dict):
    for k, v in data_dict.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                if obj.args.num_gpus:
                    v_ = v_.cuda()
                v[k_] = v_.view(-1, *v_.shape[2:])
            data_dict[k] = v
        else:
            if obj.args.num_gpus:
                v = v.cuda()
            data_dict[k] = v.view(-1, *v.shape[2:])

    return data_dict


def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou

def contrastive_loss(pos_dot, neg_dot, t=0.35, m=0.0, N=1):
    a = torch.exp((pos_dot-m)/t)
    b = torch.exp((neg_dot-m)/t)
    loss = - torch.log(a/(a+torch.sum(b, dim=0)))/N
    return torch.sum(loss, dim=0)


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


@torch.no_grad()
def visualize_data(obj, data_dict):
    b = data_dict['source_img'].shape[0] // obj.num_source_frames
    t = data_dict['target_img'].shape[0] // b
    device = data_dict['target_img'].device
    # Visualize 3d warpings

    w1 = data_dict['source_xy_warp_resize']
    w2 = data_dict['target_uv_warp_resize']
    # data_dict['source_rotation_warp_2d'] = data_dict['source_rotation_warp'].mean(1)[..., :2]
    # data_dict['source_xy_warp_resize'] = data_dict['source_xy_warp_resize'].mean(1)[..., :2]
    # data_dict['target_uv_warp_resize_2d'] = data_dict['target_uv_warp_resize'].mean(1)[..., :2]
    # data_dict['target_rotation_warp_2d'] = data_dict['target_rotation_warp'].mean(1)[..., :2]

    try:
        data_dict['source_stickman'] = obj.draw_stickman(obj.args, data_dict['source_keypoints']).to(device)
        data_dict['target_stickman'] = obj.draw_stickman(obj.args, data_dict['target_keypoints']).to(device)

        if obj.weights['landmarks']:
            data_dict['pred_landmarks'] = obj.draw_stickman(obj.args,
                                                            data_dict['pred_target_img_landmarks'] - 0.25).to(
                device)
            data_dict['target_landmarks'] = obj.draw_stickman(obj.args,
                                                              data_dict['target_img_landmarks'] - 0.25).to(device)
    except Exception as e:
        print(e)

    # data_dict['source_motion_img'] = F.grid_sample(
    #     data_dict['source_img'],
    #     data_dict['source_rotation_warp_2d'],
    #     padding_mode='reflection')
    #
    # data_dict['target_motion_img'] = F.grid_sample(
    #     F.grid_sample(
    #         data_dict['source_motion_img'],
    #         data_dict['target_uv_warp_resize_2d']),
    #     data_dict['target_rotation_warp_2d'],
    #     padding_mode='reflection')


    # print(data_dict['source_xy_warp_resize'].shape)
    # a = [w1[:, i] for i in range(w1.shape[1])]
    # print(torch.cat(a, dim=1).shape)
    # data_dict['source_xy_warp_resize'] = torch.cat(a, dim=2)
    # b = [w2[:, i] for i in range(w2.shape[1])]
    # data_dict['target_uv_warp_resize_2d'] = torch.cat(b, dim=2)

    return data_dict



@torch.no_grad()
def get_visuals(obj, data_dict):
    # print(data_dict.keys())
    b = data_dict['source_img'].shape[0] // obj.num_source_frames
    t = data_dict['target_img'].shape[0] // b

    # This function creates an output grid of visuals
    visuals_data_dict = {}
    resize = lambda img: F.interpolate(img, mode='bilinear', size=(obj.args.output_size_s2, obj.args.output_size_s2), align_corners=False)
    data_dict['s_masked'] = data_dict['source_img'] * data_dict['source_mask']
    data_dict['s_ffhq_masked'] = data_dict['source_img_ffhq'] * data_dict['source_mask_ffhq']
    data_dict['s_ffhq_masked_add'] = data_dict['s_ffhq_masked'] - resize(data_dict['s_masked'])
    data_dict['t_masked'] = data_dict['target_img'] * data_dict['target_mask']

    resize = lambda img: F.interpolate(img, mode='bilinear', size=(obj.args.output_size_s2//4, obj.args.output_size_s2//4), align_corners=False)
    resize_up = lambda img: F.interpolate(img, mode='bilinear', size=(obj.args.output_size_s2, obj.args.output_size_s2), align_corners=False)


    # data_dict['resized_pred_add'] = resize(data_dict['target_add_ffhq_pred'] * data_dict['target_add_ffhq_pred_mask'])
    # data_dict['resized_target_add'] = resize(data_dict['pred_target_add'])
    #
    # data_dict['HF_target_ffhq'] = data_dict['target_img_ffhq'] * data_dict['resized_pred_target_face_mask'] - resize_up(resize(data_dict['target_img_ffhq'] * data_dict['resized_pred_target_face_mask']))
    # data_dict['HF_pred_ffhq'] = data_dict['pred_target_img_ffhq'] * data_dict['resized_pred_target_face_mask'] - resize_up(resize(data_dict['pred_target_img_ffhq']*data_dict['resized_pred_target_face_mask']))
    # data_dict['HF_pred'] = data_dict['resized_pred_target_img'] * data_dict['resized_pred_target_face_mask'] - resize_up(resize(data_dict['resized_pred_target_img'] * data_dict['resized_pred_target_face_mask']))
    # data_dict['HF_pred_mix'] = data_dict['resized_pred_mixing_img'] * data_dict['resized_pred_mixing_mask'] - resize_up(resize(data_dict['resized_pred_mixing_img'] * data_dict['resized_pred_mixing_mask']))
    # data_dict['HF_pred_mix_ffhq'] = data_dict['pred_mixing_img_ffhq'] * data_dict['resized_pred_mixing_mask'] - resize_up(resize(data_dict['pred_mixing_img_ffhq'] * data_dict['resized_pred_mixing_mask']))
    #
    # data_dict['masked_targed_add'] = data_dict['target_add_ffhq_pred'] * data_dict['target_add_ffhq_pred_mask']

    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        else:
            continue


        visuals_data_dict[k] = v

    visuals = []

    uvs_prep = lambda x: (x.permute(0, 3, 1, 2) + 1) / 2
    segs_prep = lambda x: torch.cat([x] * 3, dim=1)
    scores_prep = lambda x: (x + 1) / 2
    confs_prep = lambda x: x / x.max(dim=2, keepdims=True)[0].max(dim=3, keepdims=True)[0]

    visuals_list = [
        ['s_masked', None],
        ['s_ffhq_masked', None],
        # ['s_ffhq_masked_add', None],

        [f't_masked', None],
        [f'target_img_ffhq', None],
        [f'resized_pred_target_img', None],
        [f'pred_target_img_ffhq', None],
        [f'HF_target_ffhq', None],


        [f'HF_pred', None],
        [f'HF_pred_ffhq', None],
        [f'target_add_ffhq_pred', None],
        [f'masked_targed_add', None],
        [f'pred_target_add', None],
        [f'resized_target_add', None],
        [f'resized_pred_add', None],
        [f'target_add_ffhq_pred_mask', None],
        [f'target_add_ffhq_pred_mask1', None],
        [f'target_add_ffhq_pred_mask2', None],
        # [f'resized_pred_target_mask', None],
        [f'resized_pred_target_face_mask', None],





        [f'resized_pred_mixing_img', None],
        [f'pred_mixing_img_ffhq', None],
        [f'HF_pred_mix', None],
        [f'HF_pred_mix_ffhq', None],
        [f'pred_mixing_add_ffhq', None],
        [f'resized_pred_mixing_img_RESIZED', None],
        [f'pred_mixing_img_ffhq_RESIZED', None],
        [f'resized_pred_mixing_mask', None],
        [f'resized_pred_mixing_face_mask', None],



    ]


    max_h = max_w = 0

    for tensor_name, preprocessing_op in visuals_list:
        visuals += misc.prepare_visual(visuals_data_dict, tensor_name, preprocessing_op)

        if len(visuals):
            h, w = visuals[-1].shape[2:]
            max_h = max(h, max_h)
            max_w = max(w, max_w)

            # Upsample all tensors to maximum size
    for i, tensor in enumerate(visuals):
        visuals[i] = F.interpolate(tensor, size=(max_h, max_w), mode='bicubic', align_corners=False)

    visuals = torch.cat(visuals, 3)  # cat w.r.t. width
    visuals = visuals.clamp(0, 1)

    return visuals


def draw_stickman(args, poses):
    ### Define drawing options ###
    edges_parts = [
        list(range(0, 17)),  # face
        list(range(17, 22)), list(range(22, 27)),  # eyebrows (right left)
        list(range(27, 31)) + [30, 33], list(range(31, 36)),  # nose
        list(range(36, 42)), list(range(42, 48)),  # right eye, left eye
        list(range(48, 60)), list(range(60, 68))]  # lips

    closed_parts = [
        False, False, False, False, False, True, True, True, True]

    colors_parts = [
        (255, 255, 255),
        (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (0, 0, 255),
        (255, 0, 255), (0, 255, 255),
        (255, 255, 0), (255, 255, 0)]

    ### Start drawing ###
    stickmen = []

    for pose in poses:
        if pose is None:
            stickmen.append(torch.zeros(3, args.image_size, args.image_size))
            continue

        if isinstance(pose, torch.Tensor):
            xy = (pose[:, :2].detach().cpu().numpy() + 1) / 2 * args.image_size

        elif pose.max() < 1.0:
            xy = pose[:, :2] * args.image_size

        else:
            xy = pose[:, :2]

        xy = xy[None, :, None].astype(np.int32)

        stickman = np.ones((args.image_size, args.image_size, 3), np.uint8)

        for edges, closed, color in zip(edges_parts, closed_parts, colors_parts):
            stickman = cv2.polylines(stickman, xy[:, edges], closed, color, thickness=2)

        stickman = torch.FloatTensor(stickman.transpose(2, 0, 1)) / 255.
        stickmen.append(stickman)

    stickmen = torch.stack(stickmen)
    stickmen = (stickmen - 0.5) * 2.

    return stickmen

class MODNET(object):
    def __init__(self) -> None:
        super(MODNET, self).__init__()
        self.modnet_pass = '/fsx/nikitadrobyshev/EmoPortraits/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
        self.modnet = MODNet(backbone_pretrained=False)


        state_dict = torch.load(self.modnet_pass, map_location='cpu')
        new_state_dict = {}
        for k in list(state_dict.keys()):
            new_k = k[7:]
            new_state_dict[new_k] = state_dict[k]

        self.modnet.load_state_dict(new_state_dict)
        self.modnet.eval()
        self.modnet = self.modnet.cuda()

    @torch.no_grad()
    def forward(self, img):

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