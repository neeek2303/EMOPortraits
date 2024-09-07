import torch
from torch import nn
import torch.nn.functional as F
from utils.prepare_for_landmark import prepare_face_for_landmarks
import numpy as np
import losses
from torchvision import transforms
import sys
sys.path.append('/fsx/nikitadrobyshev/EmoPortraits/')
from repos.MODNet.src.models.modnet import MODNet
import cv2
from matplotlib import cm
from utils.non_specific import pca_metrics
from utils import misc

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def calc_train_losses(obj, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0, iteration=0):
    losses_dict = {}
    ffhq_per_b = 0
    if mode == 'dis':
        losses_dict['dis_adversarial'] = (
                obj.weights['adversarial'] *
                obj.adversarial_loss(
                    real_scores=data_dict['real_score_dis'],
                    fake_scores=data_dict['fake_score_dis'],
                    mode='dis'))

        if obj.args.use_mix_dis and epoch>=obj.args.dis2_train_start:
            losses_dict['dis_adversarial_mix'] = (
                    obj.weights['adversarial'] *
                    obj.adversarial_loss(
                        real_scores=data_dict['real_score_dis_mix'],
                        fake_scores=data_dict['fake_score_dis_mix'],
                        mode='dis'))


    if mode == 'gen':

        resize_vol = lambda img: F.interpolate(img, mode='bilinear', size=(64, 64), align_corners=False)

        losses_dict['gen_adversarial'] = (
                obj.weights['adversarial'] *
                obj.adversarial_loss(
                    fake_scores=data_dict['fake_score_gen'],
                    mode='gen'))


        if obj.args.use_mix_dis and epoch>=obj.args.dis2_gen_train_start and iteration%obj.args.dis2_gen_train_ratio==0: 
            losses_dict['gen_adversarial_mix'] = (
                    obj.weights['adversarial'] * obj.weights['mix_gen_adversarial'] *
                    obj.adversarial_loss(
                        fake_scores=data_dict['fake_score_gen_mix'],
                        mode='gen'))


        losses_dict['feature_matching'] = (
                obj.weights['feature_matching'] *
                obj.feature_matching_loss(
                    real_features=data_dict['real_feats_gen'],
                    fake_features=data_dict['fake_feats_gen']))

        if obj.weights['l1_weight']:
            losses_dict['L1'] = obj.weights['l1_weight'] * obj.l1_loss(data_dict['pred_target_img'],
                                                                       data_dict['target_img'])

        try:
            if obj.args.save_exp_vectors:
                if iteration!=0:
                    source_pose_all = torch.load(f'{obj.exp_dir}/exp_s_{obj.rank}.pt')
                    target_pose_all = torch.load(f'{obj.exp_dir}/exp_t_{obj.rank}.pt')

                    torch.save(torch.cat([source_pose_all, data_dict['source_pose_embed'].detach().cpu()],  dim=0), f'{obj.exp_dir}/exp_s_{obj.rank}.pt')
                    torch.save(torch.cat([target_pose_all, data_dict['target_pose_embed'].detach().cpu()],  dim=0), f'{obj.exp_dir}/exp_t_{obj.rank}.pt')
                else:
                    torch.save(data_dict['source_pose_embed'].detach().cpu(), f'{obj.exp_dir}/exp_s_{obj.rank}.pt')
                    torch.save(data_dict['target_pose_embed'].detach().cpu(), f'{obj.exp_dir}/exp_t_{obj.rank}.pt')
                # torch.save(data_dict['mixing_cycle_exp'].detach().cpu(), '{self.exp_dir}/exp_m_{iteration}.pt')
        except Exception as e:
            pass
        

        # Neural expression
        if obj.args.neu_exp_l1>0: 
            losses_dict['neutral_expr_l1_loss'] = obj.weights['neutral_expr_l1_weight']*obj.l1_loss(data_dict['pred_neutral_expr_vertor'], data_dict['pred_neutral_expr_vertor'].detach()*0)

        if obj.weights['vgg19_neutral'] and (epoch==0 and iteration<200):
            data_dict['target_img_full_aling'] = F.grid_sample(data_dict['pred_target_img'].float(), data_dict['align_warp_full'].float())
            losses_dict['vgg19_neutral'], _ = obj.vgg19_loss(data_dict['pred_neutral_img'], data_dict['target_img_full_aling'].detach(),
                                                        None)
            losses_dict['vgg19_neutral'] *= obj.weights['vgg19_neutral'] #if iteration<500 else obj.weights['vgg19_neutral']/10

    
        # Matching volumes
        if obj.args.volumes_l1>0 and epoch >= obj.args.vol_loss_epoch and iteration>20:   # not (epoch==0 and iteration<0): 
            W = obj.weights['volumes_l1']

            if obj.args.vol_loss_grad>0:
                w = min((((epoch-obj.args.vol_loss_epoch+1))/obj.args.vol_loss_grad), 1)
                W = W*w
            if iteration%950==0:
                print(W)
            loss_f = obj.l1_loss if obj.args.type_of_can_vol_loss=='l1' else obj.mse_loss
            vv = data_dict['canon_volume'].detach() if obj.args.detach_cv_l1 else data_dict['canon_volume']
            losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss'] = W*loss_f(vv, data_dict['canon_volume_from_target'].detach())
            # losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_dif_min'] = -loss_f(data_dict['canon_volume'][:1], data_dict['canon_volume'][1:].detach())
            losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_dif'] = W*loss_f(data_dict['canon_volume'][:1].detach(), data_dict['canon_volume_from_target'][1:].detach())
            # if epoch>=5:
            #     losses_dict['volumes_l1_loss'] *=2
            # if epoch>=20:
            #     losses_dict['volumes_l1_loss'] *=2

            if obj.weights['volumes_pull'] or obj.weights['volumes_push']:

                b = data_dict['canon_volume'].shape[0]
                y = torch.tensor([1]*b).to(data_dict['canon_volume'].device)
                for i in range(b):
                    cos = obj.cosin_sim(data_dict['canon_volume'][i].view(1, -1), data_dict['canon_volume_from_target'].detach()[i].view(1, -1), y)
                    if i == 0:
                        losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_pull'] = obj.weights['volumes_pull'] * (cos)
                    else:
                        losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_pull'] += obj.weights['volumes_pull'] * (cos)

                losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_pull'] /= b


                
                b = data_dict['canon_volume'].shape[0]
                y = torch.tensor([-1]*b).to(data_dict['canon_volume'].device)
                for i in range(b):
                    cos = obj.cosin_sim(data_dict['canon_volume'][i].view(1, -1), data_dict['canon_volume_from_target'].roll(1, dims=0).detach()[i].view(1, -1), y)
                    if i == 0:
                        losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_push'] = obj.weights['volumes_push'] * (cos)
                    else:
                        losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_push'] += obj.weights['volumes_push'] * (cos)

                losses_dict[f'volumes_{obj.args.type_of_can_vol_loss}_loss_push'] /= b




        if obj.pred_seg:
            losses_dict['seg_loss'] = 10 * obj.seg_loss(data_dict['pred_target_seg'], data_dict['target_mask'])

        if obj.weights['gaze']: # and epoch >= obj.args.mix_losses_start:
            try:
                gl = obj.weights['gaze'] * obj.gaze_loss(data_dict['pred_target_img'], data_dict['target_img'],
                                                           data_dict['target_keypoints'][:, :, :2])
                if gl.shape != torch.Size([]):
                    print('gaze_loss returned list: ', gl)
                    losses_dict['gaze_loss'] = losses_dict['gen_adversarial']*0
                else:
                    losses_dict['gaze_loss'] = gl
            except Exception as e:
                print(e)
                print('error in gaze')
                losses_dict['gaze_loss'] = losses_dict['gen_adversarial'] * 0


        if obj.weights['vgg19']:
            if obj.args.image_additional_size is not None:
                resize = lambda img: F.interpolate(img, mode='bilinear', size=(
                    obj.args.image_additional_size, obj.args.image_additional_size), align_corners=False)
                ns = obj.args.vgg19_num_scales - (data_dict['pred_target_img'].shape[-1]//obj.args.image_additional_size)//2
                losses_dict['vgg19'], _ = obj.vgg19_loss(resize(data_dict['pred_target_img']),
                                                            resize(data_dict['target_img']), None, num_scales=ns)
                losses_dict['vgg19'] *= obj.weights['vgg19']
            else:
                losses_dict['vgg19'], _ = obj.vgg19_loss(data_dict['pred_target_img'], data_dict['target_img'],
                                                            None)
                losses_dict['vgg19'] *= obj.weights['vgg19']

        if obj.weights['perc_face_pars']:
            losses_dict['perc_face_pars_loss'], _ = obj.perc_face_pars_loss(data_dict['pred_target_img'][:1], data_dict['target_img'][:1], None)
            losses_dict['perc_face_pars_loss'] *= obj.weights['perc_face_pars']



        if (obj.args.w_eyes_loss_l1>0 or obj.args.w_mouth_loss_l1>0 or obj.args.w_ears_loss_l1>0) and epoch >= obj.args.face_parts_epoch_start:
            
            try:
                for i in range(data_dict['target_img'].shape[0]):
                    masks_gt, logits_gt, logits_gt_soft, faces = obj.face_parsing_bug.get_lips(data_dict['target_img'][i])
                    masks_s1, logits_s1, logits_pred_soft, _ = obj.face_parsing_bug.get_lips(data_dict['pred_target_img'][i], faces)

                    logits_gt_soft = logits_gt_soft.detach()
                    logits_pred_soft = logits_pred_soft.detach()

                    indx_list = []
                    
                    if obj.args.w_eyes_loss_l1>0:
                        indx_eyes =[2, 3, 4, 5]
                        # print(data_dict['target_img'][i].shape, logits_gt_soft[:, indx_eyes].shape)
                        indx_list.append([obj.args.w_eyes_loss_l1, indx_eyes])
                        # mouth_gt = torch.stack([obj.args.w_eyes_loss_l1*data_dict['target_img'][i].unsqueeze(0)*logits_gt_soft[:, map_i:map_i+1] for map_i in  indx_eyes], dim=0).sum(0)
                        # mouth_pred = torch.stack([obj.args.w_eyes_loss_l1*data_dict['pred_target_img'][i].unsqueeze(0)*logits_gt_soft[:, map_i:map_i+1] for map_i in  indx_eyes], dim=0).sum(0)

                        if i==0:
                            data_dict['eyes_mask'] =  logits_gt_soft[:, 2:3] + logits_gt_soft[:, 3:4] + logits_gt_soft[:, 4:5] + logits_gt_soft[:, 5:6]
                            losses_dict['l1_eyes'] = obj.args.w_eyes_loss_l1*obj.l1_loss(data_dict['pred_target_img']*data_dict['eyes_mask'], data_dict['target_img']*data_dict['eyes_mask'])
                        else:
                            mask = logits_gt_soft[:, 2:3] + logits_gt_soft[:, 3:4] + logits_gt_soft[:, 4:5] + logits_gt_soft[:, 5:6]
                            losses_dict['l1_eyes'] += obj.args.w_eyes_loss_l1*obj.l1_loss(data_dict['pred_target_img']*mask, data_dict['target_img']*mask)
                            data_dict['eyes_mask'] =  torch.cat([data_dict['eyes_mask'], mask], dim=0)

                        

                    if obj.args.w_mouth_loss_l1>0:
                        indx_mouth =[7, 8, 9]
                        indx_list.append([obj.args.w_mouth_loss_l1, indx_mouth])

                        if i==0:
                            
                            data_dict['mouth_mask'] =  logits_gt_soft[:, 7:8] + logits_gt_soft[:, 8:9] + logits_gt_soft[:, 9:10]
                            losses_dict['l1_mouth'] = obj.args.w_mouth_loss_l1*obj.l1_loss(data_dict['pred_target_img']*data_dict['mouth_mask'], data_dict['target_img']*data_dict['mouth_mask'])
                        else:
                            mask = logits_gt_soft[:, 7:8] + logits_gt_soft[:, 8:9] + logits_gt_soft[:, 9:10]
                            losses_dict['l1_mouth'] += obj.args.w_mouth_loss_l1*obj.l1_loss(data_dict['pred_target_img']*mask, data_dict['target_img']*mask)
                            data_dict['mouth_mask'] =  torch.cat([data_dict['mouth_mask'], mask], dim=0)

                    if obj.args.w_ears_loss_l1>0:
                        indx_ears =[11, 12]
                        indx_list.append([obj.args.w_ears_loss_l1, indx_ears])

                        if i==0:
                            
                            data_dict['ears_mask'] =  logits_gt_soft[:, 11:12] + logits_gt_soft[:, 12:13] 
                            losses_dict['l1_ears'] = obj.args.w_ears_loss_l1*obj.l1_loss(data_dict['pred_target_img']*data_dict['mouth_mask'], data_dict['target_img']*data_dict['mouth_mask'])
                        else:
                            mask = logits_gt_soft[:, 11:12] + logits_gt_soft[:, 12:13]
                            losses_dict['l1_ears'] += obj.args.w_ears_loss_l1*obj.l1_loss(data_dict['pred_target_img']*mask, data_dict['target_img']*mask)
                            data_dict['ears_mask'] =  torch.cat([data_dict['ears_mask'], mask], dim=0)
                        
                    

                data_dict['face_parts'] = data_dict['eyes_mask'] + data_dict['mouth_mask'] + data_dict['ears_mask']
                
            except Exception as e:
                print(e)

        if obj.weights['l1_vol_rgb'] and obj.args.targ_vol_loss_scale:
            losses_dict['l1_vol_rgb_tar'] = obj.l1_loss(resize_vol(data_dict['pred_tar_img_vol']), resize_vol(data_dict['target_img'])) * obj.weights['l1_vol_rgb']/2


        if obj.weights['l1_vol_rgb'] and epoch >= obj.args.start_vol_rgb:
            w = obj.weights['l1_vol_rgb'] if epoch >= 1 else obj.weights['l1_vol_rgb'] / 1
            losses_dict['l1_vol_rgb'] = obj.l1_loss(resize_vol(data_dict['pred_tar_img_vol']), resize_vol(data_dict['pred_target_img'])) * w


        if obj.weights['l1_vol_rgb_mix'] and epoch >= obj.args.start_vol_rgb:
            w = obj.weights['l1_vol_rgb'] if epoch >= 1 else obj.weights['l1_vol_rgb'] / 1
            losses_dict['l1_vol_rgb_mix'] = obj.l1_loss(resize_vol(data_dict['pred_mixing_img_vol']), resize_vol(data_dict['pred_mixing_img'])) * w


        if obj.weights['cycle_idn'] and obj.pred_cycle:
            if data_dict['target_img'].shape[0] > 1:
                losses_dict['vgg19_cycle_idn'], _ = obj.vgg19_loss(data_dict['target_img'].detach(),
                                                                   data_dict['pred_identical_cycle'], None)

                n = data_dict['source_img'].shape[0]
                t = data_dict['target_img'].shape[0]
                inputs_orig_face_aligned = F.grid_sample(
                    torch.cat([data_dict['target_img'], data_dict['pred_identical_cycle']]).float(),
                    data_dict['align_warp'].float())
                pred_target_img_face_align, target_img_align_orig = inputs_orig_face_aligned.split([n, t], dim=0)
                losses_dict['vgg19_face_cycle_idn'], _ = obj.vgg19_loss_face(pred_target_img_face_align.detach(),
                                                                             target_img_align_orig, None)

                losses_dict['vgg19_cycle_idn'] *= obj.weights['vgg19_cycle_idn']
                losses_dict['vgg19_face_cycle_idn'] *= obj.weights['vgg19_face_cycle_idn']
            else:
                losses_dict['vgg19_cycle_idn'] = losses_dict['vgg19'] * 0
                losses_dict['vgg19_face_cycle_idn'] = losses_dict['vgg19'] * 0

        if obj.weights['cycle_exp'] and obj.pred_cycle:
            if data_dict['target_img'].shape[0] > 1:
                losses_dict['vgg19_cycle_exp'], _ = obj.vgg19_loss(data_dict['target_img'].detach(),
                                                                   data_dict['cycle_mix_pred'], None)
                n = data_dict['source_img'].shape[0]
                t = data_dict['target_img'].shape[0]
                inputs_orig_face_aligned = F.grid_sample(
                    torch.cat([data_dict['target_img'], data_dict['cycle_mix_pred']]).float(),
                    data_dict['align_warp'].float())
                pred_target_img_face_align, target_img_align_orig = inputs_orig_face_aligned.split([n, t], dim=0)
                losses_dict['vgg19_face_cycle_exp'], _ = obj.vgg19_loss_face(pred_target_img_face_align.detach(),
                                                                             target_img_align_orig, None)
                losses_dict['vgg19_cycle_exp'] *= obj.weights['vgg19_cycle_exp']
                losses_dict['vgg19_face_cycle_exp'] *= obj.weights['vgg19_face_cycle_exp']
            else:
                losses_dict['vgg19_cycle_exp'] = losses_dict['vgg19'] * 0
                losses_dict['vgg19_face_cycle_exp'] = losses_dict['vgg19'] * 0


        n = data_dict['source_img'].shape[0]
        t = data_dict['target_img'].shape[0]
        inputs_orig_face_aligned = F.grid_sample(
            torch.cat([data_dict['pred_target_img'], data_dict['target_img']]).float(),
            data_dict['align_warp'].float())
        data_dict['pred_target_img_face_align'], data_dict[
            'target_img_align_orig'] = inputs_orig_face_aligned.split([n, t], dim=0)


        if obj.weights['vgg19_face']:

            if ffhq_per_b > 0:
                losses_dict['vgg19_face'], _ = obj.vgg19_loss_face(
                    data_dict['pred_target_img_face_align'][:-ffhq_per_b],
                    data_dict['target_img_align_orig'][:-ffhq_per_b], None)
                losses_dict['vgg19_face'] *= obj.weights['vgg19_face'] * (3/4)

                losses_dict['vgg19_face_ffhq'], _ = obj.vgg19_loss_face(
                    data_dict['pred_target_img_face_align'][-ffhq_per_b:],
                    data_dict['target_img_align_orig'][-ffhq_per_b:], None)
                losses_dict['vgg19_face_ffhq'] *= obj.weights['vgg19_face'] * (1/4)
            else:
                losses_dict['vgg19_face'], _ = obj.vgg19_loss_face(data_dict['pred_target_img_face_align'],
                                                                   data_dict['target_img_align_orig'], None)
                losses_dict['vgg19_face'] *= obj.weights['vgg19_face']

        if obj.weights['resnet18_fv_mix'] and epoch >= obj.args.mix_losses_start:

            # print(data_dict['mixing_img_align'].shape, data_dict['target_img_align_orig'].shape)
            # data_dict['mixing_img_align'] = F.interpolate(data_dict['mixing_img_align'], mode='bilinear', size=(256, 256), align_corners=False)
            # data_dict['target_img_align_orig'] = F.interpolate(data_dict['target_img_align_orig'], mode='bilinear', size=(256, 256), align_corners=False)


            b=obj.args.bs_resnet18_fv_mix

            m = obj.get_face_vector_resnet.forward(data_dict['mixing_img_align'][:b])
            t = obj.get_face_vector_resnet.forward(data_dict['target_img_align_orig'][:b])

            # _, im_m = obj.get_face_vector.forward(data_dict['pred_mixing_img'], crop=True, forward=False, S=0.7)
            # _, im_t = obj.get_face_vector.forward(data_dict['target_img'], crop=True, forward=False, S=0.7)
            # m = obj.get_face_vector_resnet.forward(im_m)
            # t = obj.get_face_vector_resnet.forward(im_t)

            b = data_dict['mixing_img_align'][:b].shape[0]
            y = torch.tensor([1]*b).to(data_dict['mixing_img_align'].device)

            # losses_dict['resnet18_fv_mix'] = self.weights['resnet18_fv_mix'] * (self.cosin_sim(m.view(b, -1), t.view(b, -1), y))

            for i in range(b):
                cos = obj.cosin_sim(m[i].view(1, -1), t[i].view(1, -1), y)
                # print(torch.mean(m[i]), torch.mean(t[i]), cos)
                # print(torch.mean(m[i]), torch.mean(t[i]), cos, m[i].view(1, -1).shape, m[i][:10], t[i][:10])
                if i == 0:
                    losses_dict['resnet18_fv_mix'] = obj.weights['resnet18_fv_mix'] * (cos)
                else:
                    losses_dict['resnet18_fv_mix'] += obj.weights['resnet18_fv_mix'] * (cos)

            losses_dict['resnet18_fv_mix'] /= b

        if obj.weights['vgg19_fv_mix'] and epoch >= obj.args.mix_losses_start:
            # m = self.get_face_vector_resnet.forward(data_dict['mixing_img_align'])
            # t = self.get_face_vector_resnet.forward(data_dict['target_img_align_orig'])
            m, _ = obj.get_face_vector.forward(data_dict['mixing_img_align'], False)
            t, _ = obj.get_face_vector.forward(data_dict['target_img_align_orig'], False)
            b = data_dict['mixing_img_align'].shape[0]
            y = torch.tensor([1] * b).to(data_dict['mixing_img_align'].device)
            # y = torch.tensor([1] * b).to(data_dict['mixing_img_align'].device)
            losses_dict['vgg19_fv_mix'] = obj.weights['vgg19_fv_mix'] * (
                obj.cosin_sim(m.view(b, -1), t.view(b, -1), y))
            losses_dict['vgg19_fv_mix'] /=b

        if obj.weights['face_resnet']:
            n = data_dict['source_img'].shape[0]
            t = data_dict['target_img'].shape[0]
            inputs_orig_face_aligned = F.grid_sample(
                torch.cat([data_dict['pred_target_img'], data_dict['target_img']]).float(),
                data_dict['align_warp'].float())
            data_dict['pred_target_img_face_align'], data_dict[
                'target_img_align_orig'] = inputs_orig_face_aligned.split([n, t], dim=0)
            losses_dict['loss_face_resnet'], _ = obj.face_resnet_loss(data_dict['pred_target_img_face_align'],
                                                                      data_dict['target_img_align_orig'], None)
            losses_dict['loss_face_resnet'] *= obj.weights['face_resnet']

        if obj.weights['vgg19_emotions']:
            n = data_dict['source_img'].shape[0]
            t = data_dict['target_img'].shape[0]
            inputs_orig_face_aligned = F.grid_sample(
                torch.cat([data_dict['pred_target_img'], data_dict['target_img']]).float(),
                data_dict['align_warp'].float())
            data_dict['pred_target_img_face_align'], data_dict[
                'target_img_align_orig'] = inputs_orig_face_aligned.split([n, t], dim=0)
            losses_dict['vgg19_emotions'], _ = obj.vgg19_loss_emotions(data_dict['pred_target_img_face_align'],
                                                                       data_dict['target_img_align_orig'], None)
            losses_dict['vgg19_emotions'] *= obj.weights['vgg19_emotions']

        if obj.weights['resnet18_emotions']:
            n = data_dict['source_img'].shape[0]
            t = data_dict['target_img'].shape[0]
            inputs_orig_face_aligned = F.grid_sample(
                torch.cat([data_dict['pred_target_img'], data_dict['target_img']]).float(),
                data_dict['align_warp'].float())
            data_dict['pred_target_img_face_align'], data_dict[
                'target_img_align_orig'] = inputs_orig_face_aligned.split([n, t], dim=0)
            losses_dict['resnet18_emotions'], _ = obj.resnet18_loss_emotions(
                data_dict['pred_target_img_face_align'], data_dict['target_img_align_orig'], None)
            losses_dict['resnet18_emotions'] *= obj.weights['resnet18_emotions']

        if obj.weights['landmarks']:
            with torch.no_grad():
                retina_faces = []
                for tar_img in data_dict['target_img']:
                    try:
                        face = obj.retinaface(tar_img.unsqueeze(0) * 255)[0]
                    except Exception as e:
                        print(e)
                        print('make costil face [0, 0, 0, 0, 1]')
                        face = [0, 0, 0, 0, 1]
                    retina_faces.append(face)

            data_dict['pred_target_img_face_align_retina'] = prepare_face_for_landmarks(
                data_dict['pred_target_img'], retina_faces)
            data_dict['target_img_align_orig_retina'] = prepare_face_for_landmarks(data_dict['target_img'],
                                                                                   retina_faces)

            losses_dict['landmarks'], (data_dict['pred_target_img_landmarks'], data_dict['target_img_landmarks']) = \
                obj.landmarks_loss(data_dict['pred_target_img_face_align_retina'],
                                   data_dict['target_img_align_orig_retina'], None)

            losses_dict['landmarks'] *= obj.weights['landmarks']
            data_dict['pred_target_img_landmarks'] = data_dict['pred_target_img_landmarks'].reshape(-1, 68, 2)
            data_dict['target_img_landmarks'] = data_dict['target_img_landmarks'].reshape(-1, 68, 2)

        if obj.weights['warping_reg']:
            losses_dict['warping_reg'] = obj.weights['warping_reg'] * obj.warping_reg_loss(
                data_dict['target_motion_deltas'])


        ###### barlow twince ######
        if obj.weights['barlow'] and epoch >= obj.args.contr_losses_start:
            b = data_dict['target_pose_embed'].shape[0]
            tar = data_dict['target_pose_embed'].view(b, -1)
            # m = tar.shape[1]
            # print(tar.shape, data_dict['target_pose_embed'].shape)

            losses_dict['Barlow_loss'] = losses_dict['vgg19'] * 0

            for x in [data_dict['pred_cycle_exp'], data_dict['mixing_cycle_exp']]:
                r = x.view(b, -1)
                c = obj.bn(r).T @ obj.bn(tar)
                # lambd = 0.0051
                lambd = 1/512
                # sum the cross-correlation matrix between all gpus
                c.div_(b)
                torch.distributed.all_reduce(c)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()

                losses_dict['Barlow_loss'] +=  obj.weights['barlow'] * (on_diag + lambd * off_diag)

        ###### pull_exp ###### 
        if epoch >= obj.args.contr_losses_start:
            b = data_dict['target_pose_embed'].shape[0]
            y = torch.tensor([1] * b).to(data_dict['target_pose_embed'].device)
            # mix_w = 1
            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2  # + min(50, epoch) / 50
            mix_w = mix_w if epoch <= obj.args.mix_losses_start*2 else 3 #6
            a1 = obj.cosin_sim_pos(data_dict['pred_cycle_exp'].view(b, -1), data_dict['target_pose_embed'].view(b, -1), y)
            a2 = mix_w * obj.cosin_sim_pos(data_dict['mixing_cycle_exp'].view(b, -1), data_dict['target_pose_embed'].view(b, -1), y)

            losses_dict['pull_exp'] = obj.weights['pull_exp'] * (a1 + a2)

            if obj.visualize:
                print(f'Pull Exp: pred - target: {a1}, mixing - target: {a2}')

        ###### push_exp ###### 
        if epoch >= obj.args.contr_losses_start:
            b = data_dict['target_pose_embed'].shape[0]
            y = torch.tensor([-1] * b).to(data_dict['target_pose_embed'].device)
            losses_dict['push_exp'] = losses_dict['gen_adversarial'] * 0
            c = 0

            # mix_w = 1
            if obj.prev_targets is None:
                obj.prev_targets = [data_dict['target_pose_embed']]
            else:
                # self.prev_targets = torch.cat((self.prev_targets, data_dict['target_pose_embed']), dim=0)[-4:]
                obj.prev_targets.append(data_dict['target_pose_embed'])
                obj.prev_targets = obj.prev_targets[-obj.num_b_negs:]
                for i in range(len(obj.prev_targets) - 1):
                    obj.prev_targets[i] = obj.prev_targets[i].detach()

            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 3 #2  # + min(50, epoch) / 50

            for negs in obj.prev_targets:
                for i in range(1, b):
                    losses_dict['push_exp'] += obj.weights['push_exp'] * (
                            obj.cosin_sim(data_dict['pred_cycle_exp'].view(b, -1),
                                          negs.roll(i, dims=0).view(b, -1), y) +
                            mix_w * obj.cosin_sim(data_dict['mixing_cycle_exp'].view(b, -1),
                                                  negs.roll(i, dims=0).view(b, -1), y))

            if epoch >= obj.args.mix_losses_start:
                if obj.args.separate_stm and iteration%(obj.args.sec_dataset_every//2) == 0:
                    c2 = torch.mean(obj.cosin_sim_2(data_dict['source_pose_embed'].view(b, -1).detach()[1:], data_dict['target_pose_embed'].view(b, -1)[1:], y)) #stm
                    c1 = torch.mean(obj.cosin_sim_2(data_dict['source_pose_embed'].view(b, -1).detach()[:1], data_dict['target_pose_embed'].view(b, -1)[:1], y)) #stm
                    stm_cur_w = obj.weights['stm'] #if iteration%(obj.args.sec_dataset_every//2) == 0 else 0.75
                    losses_dict['push_exp'] +=obj.weights['push_exp'] * stm_cur_w * c2 #*3 #* mix_w #* (b-1)
                    losses_dict['push_exp'] +=obj.weights['push_exp'] * 1.5 * c1 #*3 #* mix_w #* (b-1)
                else:
                    c = torch.mean(obj.cosin_sim_2(data_dict['source_pose_embed'].view(b, -1).detach(), data_dict['target_pose_embed'].view(b, -1), y)) #stm
                    stm_cur_w = obj.weights['stm'] #if iteration%(obj.args.sec_dataset_every//2) == 0 else 0.75
                    losses_dict['push_exp'] +=obj.weights['push_exp'] * stm_cur_w * c #*3 #* mix_w #* (b-1)


            # if epoch >= 30:
            #     losses_dict['push_exp']*= 1 + (epoch-30) / 100
            #     losses_dict['push_exp'] /= obj.num_b_negs + 1
            # else:
            #     losses_dict['push_exp'] /= obj.num_b_negs

            losses_dict['push_exp'] /= obj.num_b_negs

            a1, a2 = 0, 0
            for i in range(1, b):
                a1 += obj.cosin_sim(data_dict['pred_cycle_exp'].view(b, -1),
                                    data_dict['target_pose_embed'].roll(i, dims=0).view(b, -1), y)
                a2 += obj.cosin_sim(data_dict['mixing_cycle_exp'].roll(i, dims=0).view(b, -1),
                                    data_dict['target_pose_embed'].view(b, -1), y)

            # if obj.visualize:
            #     print(f'Push Exp: pred - target: {a1 / (b - 1)}, mixing - target: {a2 / (b - 1)}, source - target: {c}')

        ###### contrastive_exp ###### 
        if obj.weights['contrastive_exp'] and epoch >= obj.args.contr_losses_start:
            b = data_dict['target_pose_embed'].shape[0]
            y = torch.tensor([-1] * b).to(data_dict['target_pose_embed'].device)

            negs_pred = []
            negs_mix = []
            negs_source = []
            mix_w = 1
            # mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2
            N = (b - 1) * obj.num_b_negs

            if obj.prev_targets is None:
                obj.prev_targets = [data_dict['target_pose_embed']]

            if b > 1:
                for negs in obj.prev_targets:
                    for i in range(1, b):
                        negs_pred.append(
                            obj.cosin_dis(data_dict['pred_cycle_exp'].view(b, -1), negs.roll(i, dims=0).view(b, -1)))
                        negs_mix.append(
                            obj.cosin_dis(data_dict['mixing_cycle_exp'].view(b, -1), negs.roll(i, dims=0).view(b, -1)))

                if epoch >= obj.args.mix_losses_start and obj.weights['stm']>0.01:
                    negs_source.append(obj.cosin_sim_2(data_dict['source_pose_embed'].view(b, -1).detach(), data_dict['target_pose_embed'].view(b, -1), y))
                    N+=1

                pos_pred = obj.cosin_dis(data_dict['pred_cycle_exp'].view(b, -1),
                                         data_dict['target_pose_embed'].view(b, -1))
                pos_mix = obj.cosin_dis(data_dict['mixing_cycle_exp'].view(b, -1),
                                        data_dict['target_pose_embed'].view(b, -1))

                negs_pred = torch.stack(negs_pred + negs_source, dim=0)
                losses_dict['contrastive_exp_pred'] = obj.weights['contrastive_exp'] * contrastive_loss(pos_pred,
                                                                                                            negs_pred,
                                                                                                            t=0.2,
                                                                                                            m=0.2,
                                                                                                            N=N if obj.num_b_negs > 1 else 1)

                if epoch >= obj.args.mix_losses_start:
                    negs_mix = torch.stack(negs_mix + negs_source, dim=0)
                    losses_dict['contrastive_exp_mix'] = mix_w * obj.weights[
                        'contrastive_exp'] * contrastive_loss(pos_mix, negs_mix, t=0.2, m=0.2,
                                                                  N=N if obj.num_b_negs > 1 else 1)
                # print(negs_pred.shape, pos_pred.shape, losses_dict['contrastive_exp_pred'].shape)
                # raise ValueError
            else:
                losses_dict['contrastive_exp_pred'] = losses_dict['vgg19'] * 0
                losses_dict['contrastive_exp_mix'] = losses_dict['vgg19'] * 0

        ###### contrastive_exp ######  default: False
        if obj.weights['contrastive_idt']:
            b = data_dict['idt_embed'].shape[0]
            negs_1 = []
            negs_2 = []
            negs_3 = []
            # mix_w = 1
            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2
            if b > 1:
                for i in range(1, b):
                    negs_1.append(obj.cosin_dis(data_dict['idt_embed_face_target'].view(b, -1),
                                                data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1)))
                    negs_2.append(obj.cosin_dis(data_dict['idt_embed_face_pred'].view(b, -1),
                                                data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1)))
                    negs_3.append(obj.cosin_dis(data_dict['idt_embed_face_mix'].view(b, -1),
                                                data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1)))

                pos_1 = obj.cosin_dis(data_dict['idt_embed_face_target'].view(b, -1),
                                      data_dict['idt_embed_face'].view(b, -1))
                pos_2 = obj.cosin_dis(data_dict['idt_embed_face_pred'].view(b, -1),
                                      data_dict['idt_embed_face'].view(b, -1))
                pos_3 = obj.cosin_dis(data_dict['idt_embed_face_mix'].view(b, -1),
                                      data_dict['idt_embed_face'].view(b, -1))

                negs_1 = torch.stack(negs_1, dim=0)
                negs_2 = torch.stack(negs_2, dim=0)
                negs_3 = torch.stack(negs_3, dim=0)

                losses_dict['contrastive_idt_tar'] = obj.weights['contrastive_idt'] * contrastive_loss(pos_1,
                                                                                                           negs_1,
                                                                                                           t=0.2,
                                                                                                           m=0.2)
                losses_dict['contrastive_idt_pred'] = obj.weights['contrastive_idt'] * contrastive_loss(pos_2,
                                                                                                            negs_2,
                                                                                                            t=0.2,
                                                                                                            m=0.2)
                if epoch >= 1:
                    losses_dict['contrastive_idt_mix'] = mix_w * obj.weights[
                        'contrastive_idt'] * contrastive_loss(pos_3, negs_3, t=0.2, m=0.2)
            else:
                losses_dict['contrastive_idt'] = losses_dict['vgg19'] * 0

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


def calc_test_losses(obj, data_dict: dict, iteration=0):
    pred_dtype = data_dict['pred_target_img'].type()
    dtype = data_dict['target_img'].type()
    b = data_dict['pred_target_img'].shape[0]
    if pred_dtype != dtype:
        data_dict['pred_target_img'] = data_dict['pred_target_img'].type(dtype)

    expl_var = None
    expl_var_test = None

    with torch.no_grad():
        face_vector_target, target_face = obj.get_face_vector.forward(data_dict['target_img'])
        face_vector_mixing, mixing_face = obj.get_face_vector.forward(data_dict['pred_mixing_img'])
        face_vector_target_resnet_no_crop = obj.get_face_vector_resnet.forward(data_dict['target_img'])
        face_vector_mixing_resnet_no_crop = obj.get_face_vector_resnet.forward(data_dict['pred_mixing_img'])

        y = torch.tensor([1] * b).to(data_dict['target_img'].device)

    losses_dict = {
        'ssim': obj.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean(),
        'psnr': obj.psnr(data_dict['pred_target_img'], data_dict['target_img']),
        'lpips': obj.lpips(data_dict['pred_target_img'], data_dict['target_img']),
        'face_vgg': obj.l1_loss(face_vector_target, face_vector_mixing),
        # 'face_resnet': obj.l1_loss(face_vector_target_resnet, face_vector_mixing_resnet),
        'face_resnet_no_crop': obj.l1_loss(face_vector_target_resnet_no_crop, face_vector_mixing_resnet_no_crop),
        'face_vgg_cos': obj.cosin_sim(face_vector_target.view(b, -1), face_vector_mixing.view(b, -1), y),
        # 'face_resnet_cos': obj.cosin_sim(face_vector_target_resnet.view(b, -1), face_vector_mixing_resnet.view(b, -1),
        #                                  y),
        'face_resnet_no_crop_cos': obj.cosin_sim(face_vector_target_resnet_no_crop.view(b, -1),
                                                 face_vector_mixing_resnet_no_crop.view(b, -1), y)
    }

    try:
        if obj.args.save_exp_vectors and iteration==0:
            source_pose_all = torch.cat([torch.load(f'{obj.exp_dir}/exp_s_{i}.pt') for i in range(obj.args.num_gpus)], dim=0).view(-1, obj.args.lpe_output_channels_expression)
            target_pose_all = torch.cat([torch.load(f'{obj.exp_dir}/exp_t_{i}.pt') for i in range(obj.args.num_gpus)], dim=0).view(-1, obj.args.lpe_output_channels_expression)
            expl_var, losses_dict['auc'], losses_dict['cumsum 1'], losses_dict['cumsum 0.1'], losses_dict['cumsum 0.01']  = pca_metrics([source_pose_all, target_pose_all], obj.args.lpe_output_channels_expression)
            # expl_var, _, losses_dict['N > 1 expl_var'], losses_dict['N > 0.1 expl_var'], losses_dict['N > 0.01 expl_var']  = pca_metrics([source_pose_all, target_pose_all], obj.args.lpe_output_channels_expression)
    except Exception as e:
        pass
    # if obj.args.save_exp_vectors:
    #     if iteration!=0:
    #         source_pose_all = torch.load(f'{obj.exp_dir}/exp_s_test_{obj.rank}.pt')
    #         target_pose_all = torch.load(f'{obj.exp_dir}/exp_t_test_{obj.rank}.pt')

    #         torch.save(torch.cat([source_pose_all, data_dict['source_pose_embed'].detach().cpu()],  dim=0), f'{obj.exp_dir}/exp_s_test_{obj.rank}.pt')
    #         torch.save(torch.cat([target_pose_all, data_dict['target_pose_embed'].detach().cpu()],  dim=0), f'{obj.exp_dir}/exp_t_test_{obj.rank}.pt')
    #     else:
    #         torch.save(data_dict['source_pose_embed'].detach().cpu(), f'{obj.exp_dir}/exp_s_test_{obj.rank}.pt')
    #         torch.save(data_dict['target_pose_embed'].detach().cpu(), f'{obj.exp_dir}/exp_t_test_{obj.rank}.pt')

    # try:
    #     if obj.args.save_exp_vectors and iteration==-1:
    #         source_pose_all = torch.cat([torch.load(f'{obj.exp_dir}/exp_s_test_{i}.pt') for i in range(obj.args.num_gpus)], dim=0).view(-1, obj.args.lpe_output_channels_expression)
    #         target_pose_all = torch.cat([torch.load(f'{obj.exp_dir}/exp_t_test_{i}.pt') for i in range(obj.args.num_gpus)], dim=0).view(-1, obj.args.lpe_output_channels_expression)
    #         expl_var_test, losses_dict['AUC_expl_var_test'], losses_dict['N > 1 expl_var_test'], losses_dict['N > 0.1 expl_var_test'], losses_dict['N > 0.01 expl_var_test']  = pca_metrics([source_pose_all, target_pose_all], obj.args.lpe_output_channels_expression)
    # except Exception as e:
    #     pass

    

    if obj.sep_test_losses:
        losses_dict['ssim person'] = obj.ssim(data_dict['pred_target_img'] * data_dict['target_mask'],
                                              data_dict['target_img'] * data_dict['target_mask']).mean()
        losses_dict['psnr person'] = obj.psnr(data_dict['pred_target_img'] * data_dict['target_mask'],
                                              data_dict['target_img'] * data_dict['target_mask'])
        losses_dict['lpips person'] = obj.lpips(data_dict['pred_target_img'] * data_dict['target_mask'],
                                                data_dict['target_img'] * data_dict['target_mask'])
        losses_dict['ssim back'] = obj.ssim(data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
                                            data_dict['target_img'] * (1 - data_dict['target_mask'])).mean()
        losses_dict['psnr back'] = obj.psnr(data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
                                            data_dict['target_img'] * (1 - data_dict['target_mask']))
        losses_dict['lpips back'] = obj.lpips(data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
                                              data_dict['target_img'] * (1 - data_dict['target_mask']))

    if obj.args.image_size > 160:
        losses_dict['ms_ssim'] = obj.ms_ssim(data_dict['pred_target_img'], data_dict['target_img']).mean()
        if obj.sep_test_losses:
            losses_dict['ms_ssim person'] = obj.ms_ssim(data_dict['pred_target_img'] * data_dict['target_mask'],
                                                        data_dict['target_img'] * data_dict['target_mask']).mean()
            losses_dict['ms_ssim back'] = obj.ms_ssim(
                data_dict['pred_target_img'] * (1 - data_dict['target_mask']),
                data_dict['target_img'] * (1 - data_dict['target_mask'])).mean()
    

    return losses_dict, expl_var, expl_var_test


def init_losses(obj, args):
    if obj.weights['adversarial']:
        obj.adversarial_loss = losses.AdversarialLoss()

    if obj.weights['feature_matching']:
        obj.feature_matching_loss = losses.FeatureMatchingLoss()

    if obj.weights['gaze']:
        obj.gaze_loss = losses.GazeLoss(device='cuda', gaze_model_types=['vgg16'], project_dir = obj.args.project_dir) #weights=(0.0, 0.0, 0.0, 0.0, 1.0) , weights=(0.3, 0.25, 0.2, 0.15, 0.1)

    if obj.weights['vgg19']:
        obj.vgg19_loss = losses.PerceptualLoss(num_scales=args.vgg19_num_scales, use_fp16=False)

    if obj.weights['vgg19_face']:
        obj.vgg19_loss_face = losses.PerceptualLoss(num_scales=2, network='vgg_face_dag',
                                                    layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                                    resize=True, weights=(0.03125, 0.0625, 0.125, 0.25, 1.0), use_fp16=False)

    if obj.weights['perc_face_pars']:
        obj.perc_face_pars_loss = losses.PerceptualLoss(num_scales=2, network='face_parsing',
                                                    layers=['feat4', 'feat8', 'feat16', 'feat32'],
                                                    resize=True, weights=(0.0625, 0.125, 0.25, 1.0), use_fp16=False)
        
    if obj.weights['face_resnet']:
        obj.face_resnet_loss= losses.PerceptualLoss(num_scales=1, network='face_resnet',
                                                    layers=['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12', 'relu13', 'relu14', 'relu15', 'relu16'],
                                                    resize=True, weights=(0.03125, 0.03125, 0.03125, 0.0625, 0.0625, 0.0625, 0.125, 0.125, 0.125, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0),
                                                    apply_normalization=False, face_norm=True)

    if obj.weights['vgg19_emotions']:
        obj.vgg19_loss_emotions = losses.PerceptualLoss(num_scales=2, network='vgg_emotions', resize=True,
                                                        gray=True, weights=(0.2, 0.2, 0.2, 0.2, 0.2),
                                                        resize_size=55, scale_factor=0.8,
                                                        apply_normalization=False)

    if obj.weights['resnet18_emotions']:
        obj.resnet18_loss_emotions = losses.PerceptualLoss(num_scales=1, network='resnet_emotions', resize=True,
                                                           layers=['layer_1', 'layer_2', 'layer_3', 'layer_4'],
                                                           weights=(0.25, 0.25, 0.25, 0.25))

    if obj.weights['landmarks']:
        obj.landmarks_loss = losses.PerceptualLoss(num_scales=1, network='landmarks', resize=False,
                                                   apply_normalization=False,
                                                   layers=['conv1', 'conv2_dw', 'conv_23', 'conv_3', 'conv_34',
                                                            'conv_4', 'conv_45', 'conv_5', 'conv_6_sep',
                                                            'output_layer'],
                                                   weights=(0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.3, 0.8, 1, 1))

    if obj.weights['warping_reg']:
        obj.warping_reg_loss = losses.WarpingRegLoss()

    obj.l1_loss = nn.L1Loss()
    obj.mse_loss = nn.MSELoss()
    obj.cosin_sim_pos = torch.nn.CosineEmbeddingLoss(margin=0.1)
    obj.cosin_sim_pos_0005 = torch.nn.CosineEmbeddingLoss(margin=0.005)
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
            try:
                if obj.args.num_gpus:
                    v = v.cuda()
                
                data_dict[k] = v.view(-1, *v.shape[2:])
            except Exception as e:
                print(e)
                print(k, v.shape)

    return data_dict


def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou

def contrastive_loss(pos_dot, neg_dot, t=0.35, m=0.0, N=1):
    a = torch.exp((pos_dot-m)/t)
    b = torch.exp((neg_dot)/t)
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
    data_dict['source_rotation_warp_2d'] = data_dict['source_rotation_warp'].mean(1)[..., :2]
    data_dict['source_xy_warp_resize_2d'] = data_dict['source_xy_warp_resize'].mean(1)[..., :2]
    data_dict['target_uv_warp_resize_2d'] = data_dict['target_uv_warp_resize'].mean(1)[..., :2]
    data_dict['target_rotation_warp_2d'] = data_dict['target_rotation_warp'].mean(1)[..., :2]


    data_dict['source_xy_warp_resize_2d'] = F.interpolate(data_dict['source_xy_warp_resize_2d'][:, :, :, 0].unsqueeze(1), size=(512, 512), mode='bicubic', align_corners=False)
    data_dict['target_uv_warp_resize_2d'] = F.interpolate(data_dict['target_uv_warp_resize_2d'][:, :, :, 0].unsqueeze(1), size=(512, 512), mode='bicubic', align_corners=False)

    data_dict['source_xy_warp_resize_2d'] = (data_dict['source_xy_warp_resize_2d']+1)/2
    data_dict['target_uv_warp_resize_2d'] = (data_dict['target_uv_warp_resize_2d']+1)/2

    try:
        data_dict['source_stickman'] = obj.draw_stickman(obj.args, data_dict['source_keypoints']).to(device)
        data_dict['source_stickman_warped'] = obj.draw_stickman(obj.args, data_dict['source_warped_keypoints']).to(device)
        data_dict['source_stickman_warped_aligned'] = obj.draw_stickman(obj.args, data_dict['source_warped_keypoints_n']).to(device)
        # data_dict['source_stickman_nn_a'] = obj.draw_stickman(obj.args, data_dict['source_warped_keypoints_nn']).to(device)
        
        
        data_dict['target_stickman'] = obj.draw_stickman(obj.args, data_dict['target_keypoints']).to(device)
        # if 'target_warped_keypoints' in list(data_dict.keys()):
        #     data_dict['target_stickman_warped'] = obj.draw_stickman(obj.args, data_dict['target_warped_keypoints']).to(device)
        #     data_dict['target_stickman_pre'] = obj.draw_stickman(obj.args, data_dict['target_pre_warped_keypoints']).to(device)
            # data_dict['target_stickman_n_aligned_b'] = obj.draw_stickman(obj.args, data_dict['target_warped_keypoints_aligned_b']).to(device)
        

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

    # data_dict['target_motion_img'] = F.grid_sample(
    #     F.grid_sample(
    #         data_dict['source_motion_img'],
    #         data_dict['target_uv_warp_resize_2d']),
    #     data_dict['target_rotation_warp_2d'],
    #     padding_mode='reflection')

    return data_dict

def depth_col(x, relative=False):
    tx = x[:,0,:,:].numpy()
    magma = cm.get_cmap('RdYlBu')
    tens_list = [tx[i] for i in range(x.shape[0])]
    if relative:
        new_list = []
        for i in tens_list:
            max = np.amax(i)
            min = np.amin(i)
            i = (i - min)*(1/(max-min + 0.0001))
            new_list.append(i)
        tens_list = new_list
    x_transformed = torch.tensor(list(map(magma, tens_list))).permute(0, 3, 1, 2)
    return x_transformed[:, :3, :, :]

@torch.no_grad()
def get_visuals(obj, data_dict):
    # print(data_dict.keys())
    b = data_dict['source_img'].shape[0] // obj.num_source_frames
    t = data_dict['target_img'].shape[0] // b

    # This function creates an output grid of visuals
    visuals_data_dict = {}

    data_dict['s_masked'] = data_dict['source_img'] * data_dict['source_mask']
    data_dict['sb_masked'] = data_dict['source_img'] * (1 - data_dict['source_mask'])
    data_dict['t_masked'] = data_dict['target_img'] * data_dict['target_mask']
    data_dict['target_mask_p'] = data_dict['target_mask']
    




    if 'face_parts' in list(data_dict.keys()):
        data_dict['face_parts_mask'] = (data_dict['face_parts']).clamp(0, 1)
    if data_dict.get('pred_mixing_depth_vol')!=None:
        data_dict['pred_mixing_depth_vol'] = (data_dict['pred_mixing_depth_vol']+1)/2
        data_dict['pred_tar_depth_vol'] = (data_dict['pred_tar_depth_vol'] + 1) / 2

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
    depth_col_rel = lambda x: depth_col(x, relative=True)

    visuals_list = [
        ['source_stickman', None],
        ['source_img', None],
        ['source_mask_modnet', None],
        ['source_mask_face_pars', None],
        ['pre_ready_mask_sou', None],
        ['source_mask_s', None],
        ['source_mask', None],
        # ['source_rotation_warp_2d', uvs_prep],
        # ['source_xy_warp_resize', uvs_prep],
        ['source_warp_aug', None],
        ['source_img_align', None],
        ['s_masked', None],
        ['sb_masked', None],
        ['t_masked', None],
        [f'target_face_img', None],
        [f'target_img_to_dis', None],
        [f'source_stickman', None],
        [f'source_stickman_warped', None],
        [f'source_stickman_warped_aligned', None],


        # [f'target_stickman_n_aligned_b', None],
        
        
        
        [f'target_img', None],
        [f'target_stickman_pre', None],
        [f'target_stickman_warped', None],
        [f'target_stickman', None],


        #############################################
        [f'pred_target_img', None],
        [f'pred_mixing_img', None],
        [f'pred_neutral_img', None],
        [f'target_img_full_aling', None],
        [f'pred_neutral_img_aligned', None],
        ['rolled_mix', None],
        [f'pred_target_img_face_align', None],
        # [f'pred_target_img_face_align_retina_{i}', None],
        [f'target_img_align_orig', None],
        [f'pred_target_seg', None],
        [f'target_mask_p', None],
        [f'face_parts_mask', None],
        [f'source_xy_warp_resize_2d', None],
        [f'target_uv_warp_resize_2d', None],
        
        # [f'target_img_align_orig_retina_{i}', None],
        # [f'pred_target_img_flip_{i}', None],
        # [f'target_stickman', None],
        [f'target_landmarks', None],
        [f'target_warp_aug', None],
        ['pred_landmarks', None],
        ['mixing_img_align', None],
        ['target_img_align_orig',None],
        ['pred_mixing_mask', None],
        ['pred_mixing_seg', None],

        #############################################

        ['pred_identical_cycle', None],
        # ['rolled_mix', None],
        ['rolled_mix_align', None],
        ['cycle_mix_pred', None],
        ['pred_mixing_seg', None],

        ['pred_tar_img_vol', None],
        ['pred_tar_depth_vol', None],
        ['pred_tar_depth_vol', depth_col],
        ['pred_tar_depth_vol', depth_col_rel],
        ['pred_mixing_img_vol', None],
        ['pred_mixing_depth_vol', None],
        ['pred_mixing_depth_vol', depth_col],
        ['pred_mixing_depth_vol', depth_col_rel],

    ]


    max_h = max_w = 0

    for tensor_name, preprocessing_op in visuals_list:
        visuals += misc.prepare_visual(visuals_data_dict, tensor_name, preprocessing_op)

        if len(visuals):
            try:
                h, w = visuals[-1].shape[2:]
                max_h = max(h, max_h)
                max_w = max(w, max_w)
            except Exception as e:
                print(tensor_name, visuals.shape)
                print(visuals_list)

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
    def __init__(self, project_dir) -> None:
        super(MODNET, self).__init__()
        self.modnet_pass = f'{project_dir}/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
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