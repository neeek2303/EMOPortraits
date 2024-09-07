import torch
from torch import nn
import torch.nn.functional as F
from utils.prepare_for_landmark import prepare_face_for_landmarks
import numpy as np
import losses
from torchvision import transforms
from .repos.MODNet.src.models.modnet import MODNet
import cv2
from utils import misc

def calc_train_losses(obj, data_dict: dict, mode: str = 'gen', epoch=0, ffhq_per_b=0):
    losses_dict = {}

    if mode == 'dis':
        losses_dict['dis_adversarial'] = (
                obj.weights['adversarial'] *
                obj.adversarial_loss(
                    real_scores=data_dict['real_score_dis'],
                    fake_scores=data_dict['fake_score_dis'],
                    mode='dis'))

        if obj.args.use_hq_disc:
            losses_dict['dis_adversarial_2'] = (
                    obj.weights['adversarial'] *
                    obj.adversarial_loss(
                        real_scores=data_dict['real_score_dis_2'],
                        fake_scores=data_dict['fake_score_dis_2'],
                        mode='dis'))

    if mode == 'gen':
        losses_dict['gen_adversarial'] = (
                obj.weights['adversarial'] *
                obj.adversarial_loss(
                    fake_scores=data_dict['fake_score_gen'],
                    mode='gen'))

        if obj.args.use_hq_disc:
            losses_dict['gen_adversarial_2'] = (obj.weights['adversarial'] *
                                                       obj.adversarial_loss(
                                                           fake_scores=data_dict['fake_score_gen_2'],
                                                           mode='gen'))

        losses_dict['feature_matching'] = (
                obj.weights['feature_matching'] *
                obj.feature_matching_loss(
                    real_features=data_dict['real_feats_gen'],
                    fake_features=data_dict['fake_feats_gen']))

        if obj.args.use_hq_disc:
            losses_dict['feature_matching_2'] = (
                    obj.weights['feature_matching'] *
                    obj.feature_matching_loss(
                        real_features=data_dict['real_feats_gen_2'][-ffhq_per_b:],
                        fake_features=data_dict['fake_feats_gen_2'][-ffhq_per_b:]))

        if obj.pred_seg:
            losses_dict['seg_loss'] = 10 * obj.seg_loss(data_dict['pred_target_seg'], data_dict['target_mask'])

        if obj.weights['gaze']:
            try:
                gl = obj.weights['gaze'] * obj.gaze_loss(data_dict['pred_target_img'], data_dict['target_img'],
                                                           data_dict['target_keypoints'][:, :, :2])
                if gl.shape != torch.Size([]):
                    print('gaze_loss returned list: ', gl)
                else:
                    losses_dict['gaze_loss'] = gl
            except Exception as e:
                print(e)
                print('error in gaze')
                # raise ValueError
                losses_dict['gaze_loss'] = losses_dict['gen_adversarial'] * 0
                # raise ValueError('error in gaze')

        # data_dict['source_img'] = data_dict['source_img'].requires_grad_(True)
        # data_dict['source_img'].retain_grad()

        if obj.dec_pred_conf:
            losses_dict['vgg19'], losses_dict['vgg19_conf'] = obj.vgg19_loss(data_dict['pred_target_img'],
                                                                             data_dict['target_img'],
                                                                             data_dict['target_vgg19_conf_ms'])
            losses_dict['vgg19'], losses_dict['vgg19_conf'] = obj.weights['vgg19'] * losses_dict['vgg19'], \
                                                              obj.weights['vgg19'] * losses_dict['vgg19_conf']
            if obj.pred_flip:
                losses_dict['vgg19_flip'], losses_dict['vgg19_conf_flip'] = obj.vgg19_loss(
                    data_dict['pred_target_img_flip'], data_dict['target_img'],
                    data_dict['target_vgg19_conf_flip_ms'])
                losses_dict['vgg19_flip'], losses_dict['vgg19_conf_flip'] = obj.weights['vgg19'] * losses_dict[
                    'vgg19_flip'], obj.weights['vgg19'] * losses_dict['vgg19_conf_flip']
        else:

            if ffhq_per_b > 0:
                resize = lambda img: F.interpolate(img, mode='bilinear', size=(
                    obj.args.image_additional_size, obj.args.image_additional_size), align_corners=False)
                ns = obj.args.vgg19_num_scales - data_dict['pred_target_img'].shape[-1]//obj.args.image_additional_size//2
                losses_dict['vgg19'], _ = obj.vgg19_loss(resize(data_dict['pred_target_img'][:-ffhq_per_b]),
                                                         resize(data_dict['target_img'][:-ffhq_per_b]), None,
                                                         num_scales=ns)
                losses_dict['vgg19'] *= obj.weights['vgg19'] * (3 / 4)

                losses_dict['vgg19_ffhq'], _ = obj.vgg19_loss(data_dict['pred_target_img'][-ffhq_per_b:],
                                                               data_dict['target_img'][-ffhq_per_b:], None,
                                                              num_scales=obj.args.vgg19_num_scales)
                losses_dict['vgg19_ffhq'] *= obj.weights['vgg19'] * (1 / 4)
            else:
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

        if obj.weights['cycle_idn'] and not obj.only_cycle_embed:
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

        if obj.weights['cycle_exp'] and not obj.only_cycle_embed:
            if data_dict['target_img'].shape[0] > 1:
                losses_dict['vgg19_cycle_exp'], _ = obj.vgg19_loss(data_dict['target_img'].detach(),
                                                                   data_dict['pred_expression_cycle'], None)
                n = data_dict['source_img'].shape[0]
                t = data_dict['target_img'].shape[0]
                inputs_orig_face_aligned = F.grid_sample(
                    torch.cat([data_dict['target_img'], data_dict['pred_expression_cycle']]).float(),
                    data_dict['align_warp'].float())
                pred_target_img_face_align, target_img_align_orig = inputs_orig_face_aligned.split([n, t], dim=0)
                losses_dict['vgg19_face_cycle_exp'], _ = obj.vgg19_loss_face(pred_target_img_face_align.detach(),
                                                                             target_img_align_orig, None)
                losses_dict['vgg19_cycle_exp'] *= obj.weights['vgg19_cycle_exp']
                losses_dict['vgg19_face_cycle_exp'] *= obj.weights['vgg19_face_cycle_exp']
            else:
                losses_dict['vgg19_cycle_exp'] = losses_dict['vgg19'] * 0
                losses_dict['vgg19_face_cycle_exp'] = losses_dict['vgg19'] * 0

        if obj.weights['vgg19_face']:
            n = data_dict['source_img'].shape[0]
            t = data_dict['target_img'].shape[0]
            inputs_orig_face_aligned = F.grid_sample(
                torch.cat([data_dict['pred_target_img'], data_dict['target_img']]).float(),
                data_dict['align_warp'].float())
            data_dict['pred_target_img_face_align'], data_dict[
                'target_img_align_orig'] = inputs_orig_face_aligned.split([n, t], dim=0)

            # print(data_dict['pred_target_img'].shape, data_dict['pred_mixing_img'].shape,  data_dict['pred_target_img_face_align'].shape)
            # data_dict['pred_target_img_face_align'] = F.interpolate(data_dict['pred_target_img_face_align'], mode='bilinear', size=(256, 256), align_corners=False)
            # data_dict['target_img_align_orig'] = F.interpolate(data_dict['target_img_align_orig'], mode='bilinear', size=(256, 256), align_corners=False)

            if ffhq_per_b > 0:
                # resize = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.image_additional_size, self.args.image_additional_size), align_corners=False)
                losses_dict['vgg19_face'], _ = obj.vgg19_loss_face(
                    data_dict['pred_target_img_face_align'][:-ffhq_per_b],
                    data_dict['target_img_align_orig'][:-ffhq_per_b], None)
                losses_dict['vgg19_face'] *= obj.weights['vgg19_face'] * (3/4)

                losses_dict['vgg19_face_ffhq'], _ = obj.vgg19_loss_face(
                    data_dict['pred_target_img_face_align'][-ffhq_per_b:],
                    data_dict['target_img_align_orig'][-ffhq_per_b:], None)
                losses_dict['vgg19_face_ffhq'] *= obj.weights['vgg19_face'] * (1/4)
            else:
                # if self.args.image_additional_size is not None:
                #     resize = lambda img: F.interpolate(img, mode='bilinear', size=(self.args.image_additional_size, self.args.image_additional_size), align_corners=False)
                #     losses_dict['vgg19_face'], _ = self.vgg19_loss_face(resize(data_dict['pred_target_img_face_align']), resize(data_dict['target_img_align_orig']), None)
                #     losses_dict['vgg19_face'] *= self.weights['vgg19_face']
                # else:
                losses_dict['vgg19_face'], _ = obj.vgg19_loss_face(data_dict['pred_target_img_face_align'],
                                                                   data_dict['target_img_align_orig'], None)
                losses_dict['vgg19_face'] *= obj.weights['vgg19_face']

        if obj.weights['resnet18_fv_mix'] and epoch >= obj.args.mix_losses_start:

            # print(data_dict['mixing_img_align'].shape, data_dict['target_img_align_orig'].shape)
            # data_dict['mixing_img_align'] = F.interpolate(data_dict['mixing_img_align'], mode='bilinear', size=(256, 256), align_corners=False)
            # data_dict['target_img_align_orig'] = F.interpolate(data_dict['target_img_align_orig'], mode='bilinear', size=(256, 256), align_corners=False)
            m = obj.get_face_vector_resnet.forward(data_dict['mixing_img_align'])
            t = obj.get_face_vector_resnet.forward(data_dict['target_img_align_orig'])
            # m, _ = self.get_face_vector.forward(data_dict['mixing_img_align'], False)
            # t, _ = self.get_face_vector.forward(data_dict['target_img_align_orig'], False)
            b = data_dict['mixing_img_align'].shape[0]
            y = torch.tensor(1).to(data_dict['mixing_img_align'].device)

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
            y = torch.tensor([1] * b).to(data_dict['mixing_img_align'])
            # y = torch.tensor([1] * b).to(data_dict['mixing_img_align'].device)
            losses_dict['vgg19_fv_mix'] = obj.weights['vgg19_fv_mix'] * (
                obj.cosin_sim(m.view(b, -1), t.view(b, -1), y))

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

        if obj.weights['l1_weight']:
            losses_dict['L1'] = obj.weights['l1_weight'] * obj.l1_loss(data_dict['pred_target_img'],
                                                                       data_dict['target_img'])

        if obj.weights['pull_idt'] and epoch >= obj.args.contr_losses_start:
            b = data_dict['idt_embed'].shape[0]
            y = torch.tensor([1] * b).to(data_dict['idt_embed'].device)
            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2.5
            a1 = obj.cosin_sim(data_dict['idt_embed_face'].view(b, -1), data_dict['idt_embed_face_target'].view(b, -1),
                               y)
            a2 = obj.cosin_sim(data_dict['idt_embed_face'].view(b, -1), data_dict['idt_embed_face_pred'].view(b, -1),
                               y)
            a3 = obj.cosin_sim(data_dict['idt_embed_face'].view(b, -1), data_dict['idt_embed_face_mix'].view(b, -1), y)
            losses_dict['pull_idt'] = obj.weights['pull_idt'] * (a1 + a2 + mix_w * a3)

            b1 = obj.l1_loss(data_dict['idt_embed_face'].view(b, -1), data_dict['idt_embed_face_target'].view(b, -1))
            b2 = obj.l1_loss(data_dict['idt_embed_face'].view(b, -1), data_dict['idt_embed_face_pred'].view(b, -1))
            b3 = obj.l1_loss(data_dict['idt_embed_face'].view(b, -1), data_dict['idt_embed_face_mix'].view(b, -1))
            if obj.visualize:
                print(f'Pull Idt: source - target: {a1}, {b1}, source - pred: {a2}, {b2}, source - mixing: {a3}, {b3}')

        if obj.weights['push_idt'] and epoch >= obj.args.contr_losses_start:
            b = data_dict['idt_embed'].shape[0]
            y = torch.tensor([-1] * b).to(data_dict['idt_embed'].device)
            losses_dict['push_idt'] = losses_dict['gen_adversarial'] * 0

            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2.5
            for i in range(1, b):
                losses_dict['push_idt'] += obj.weights['push_idt'] * (
                        obj.cosin_sim(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                      data_dict['idt_embed_face_target'].view(b, -1), y) +
                        obj.cosin_sim(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                      data_dict['idt_embed_face_pred'].view(b, -1), y) +
                        mix_w * obj.cosin_sim(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                              data_dict['idt_embed_face_mix'].view(b, -1), y))

            a1, a2, a3 = 0, 0, 0
            for i in range(1, b):
                a1 += obj.cosin_sim(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                    data_dict['idt_embed_face_target'].view(b, -1), y)
                a2 += obj.cosin_sim(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                    data_dict['idt_embed_face_pred'].view(b, -1), y)
                a3 += obj.cosin_sim(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                    data_dict['idt_embed_face_mix'].view(b, -1), y)

            b1, b2, b3 = 0, 0, 0
            for i in range(1, b):
                b1 += obj.l1_loss(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                  data_dict['idt_embed_face_target'].view(b, -1))
                b2 += obj.l1_loss(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                  data_dict['idt_embed_face_pred'].view(b, -1))
                b3 += obj.l1_loss(data_dict['idt_embed_face'].roll(i, dims=0).view(b, -1),
                                  data_dict['idt_embed_face_mix'].view(b, -1))

            if obj.visualize:
                print(
                    f'Push Idt: source - target: {a1 / (b - 1)}, {b1 / (b - 1)} source - pred: {a2 / (b - 1)}, {b2 / (b - 1)} source - mixing: {a3 / (b - 1)}, {b3 / (b - 1)}')

        if obj.weights['pull_exp'] and epoch >= obj.args.contr_losses_start:
            b = data_dict['target_pose_embed'].shape[0]
            y = torch.tensor([1] * b).to(data_dict['target_pose_embed'].device)
            # mix_w = 1
            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2  # + min(50, epoch) / 50

            a1 = obj.cosin_sim(data_dict['pred_cycle_exp'].view(b, -1), data_dict['target_pose_embed'].view(b, -1), y)
            a2 = mix_w * obj.cosin_sim(data_dict['mixing_cycle_exp'].view(b, -1),
                                       data_dict['target_pose_embed'].view(b, -1), y)

            losses_dict['pull_exp'] = obj.weights['pull_exp'] * (a1 + a2)

            if obj.visualize:
                print(f'Pull Exp: pred - target: {a1}, mixing - target: {a2}')

        if obj.weights['push_exp'] and epoch >= obj.args.contr_losses_start:
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

            mix_w = 0.5 if epoch <= obj.args.mix_losses_start else 2  # + min(50, epoch) / 50

            for negs in obj.prev_targets:
                for i in range(1, b):
                    losses_dict['push_exp'] += obj.weights['push_exp'] * (
                            obj.cosin_sim(data_dict['pred_cycle_exp'].view(b, -1),
                                          negs.roll(i, dims=0).view(b, -1), y) +
                            mix_w * obj.cosin_sim(data_dict['mixing_cycle_exp'].view(b, -1),
                                                  negs.roll(i, dims=0).view(b, -1), y))

            if epoch >= obj.args.mix_losses_start:
                c = torch.mean(obj.cosin_sim_2(data_dict['source_pose_embed'].view(b, -1).detach(), data_dict['target_pose_embed'].view(b, -1), y))
                losses_dict['push_exp'] +=obj.weights['push_exp'] * c

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

            if obj.visualize:
                print(f'Push Exp: pred - target: {a1 / (b - 1)}, mixing - target: {a2 / (b - 1)}, source - target: {c}')

        if obj.weights['contrastive_exp'] and epoch >= obj.args.contr_losses_start:
            b = data_dict['target_pose_embed'].shape[0]
            y = torch.tensor([-1] * b).to(data_dict['target_pose_embed'].device)

            negs_pred = []
            negs_mix = []
            negs_source = []
            mix_w = 1
            # mix_w = 0.5 if epoch <= self.args.mix_losses_start else 2
            N = (b - 1) * obj.num_b_negs

            if obj.prev_targets is None:
                obj.prev_targets = [data_dict['target_pose_embed']]

            if b > 1:
                for negs in obj.prev_targets:
                    for i in range(1, b):
                        # negs_pred.append(self.cosin_sim(data_dict['pred_cycle_exp'].view(b, -1), data_dict['target_pose_embed'].roll(i, dims=0).view(b, -1), y))
                        # negs_mix.append(self.cosin_sim(data_dict['mixing_cycle_exp'].view(b, -1), data_dict['target_pose_embed'].roll(i, dims=0).view(b, -1), y))

                        negs_pred.append(
                            obj.cosin_dis(data_dict['pred_cycle_exp'].view(b, -1), negs.roll(i, dims=0).view(b, -1)))
                        negs_mix.append(
                            obj.cosin_dis(data_dict['mixing_cycle_exp'].view(b, -1), negs.roll(i, dims=0).view(b, -1)))

                if epoch >= obj.args.mix_losses_start:
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
                losses_dict['contrastive_exp'] = losses_dict['vgg19'] * 0

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


def calc_test_losses(obj, data_dict: dict):
    pred_dtype = data_dict['pred_target_img'].type()
    dtype = data_dict['target_img'].type()
    b = data_dict['pred_target_img'].shape[0]
    if pred_dtype != dtype:
        data_dict['pred_target_img'] = data_dict['pred_target_img'].type(dtype)

    with torch.no_grad():
        face_vector_target, target_face = obj.get_face_vector.forward(data_dict['target_img'])
        face_vector_mixing, mixing_face = obj.get_face_vector.forward(data_dict['pred_mixing_img'])
        face_vector_target_resnet = obj.get_face_vector_resnet.forward(target_face)
        face_vector_mixing_resnet = obj.get_face_vector_resnet.forward(mixing_face)

        face_vector_target_resnet_no_crop = obj.get_face_vector_resnet.forward(data_dict['target_img'])
        face_vector_mixing_resnet_no_crop = obj.get_face_vector_resnet.forward(data_dict['pred_mixing_img'])

    y = torch.tensor([1] * b).to(data_dict['target_img'].device)

    losses_dict = {
        'ssim': obj.ssim(data_dict['pred_target_img'], data_dict['target_img']).mean(),
        'psnr': obj.psnr(data_dict['pred_target_img'], data_dict['target_img']),
        'lpips': obj.lpips(data_dict['pred_target_img'], data_dict['target_img']),
        'face_vgg': obj.l1_loss(face_vector_target, face_vector_mixing),
        'face_resnet': obj.l1_loss(face_vector_target_resnet, face_vector_mixing_resnet),
        'face_resnet_no_crop': obj.l1_loss(face_vector_target_resnet_no_crop, face_vector_mixing_resnet_no_crop),
        'face_vgg_cos': obj.cosin_sim(face_vector_target.view(b, -1), face_vector_mixing.view(b, -1), y),
        'face_resnet_cos': obj.cosin_sim(face_vector_target_resnet.view(b, -1), face_vector_mixing_resnet.view(b, -1),
                                         y),
        'face_resnet_no_crop_cos': obj.cosin_sim(face_vector_target_resnet_no_crop.view(b, -1),
                                                 face_vector_mixing_resnet_no_crop.view(b, -1), y)
    }

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
    return losses_dict


def init_losses(obj, args):
    if obj.weights['adversarial']:
        obj.adversarial_loss = losses.AdversarialLoss()

    if obj.weights['feature_matching']:
        obj.feature_matching_loss = losses.FeatureMatchingLoss()

    if obj.weights['gaze']:
        obj.gaze_loss = losses.GazeLoss(device='cuda', gaze_model_types=['vgg16'], weights=(0.01666, 0.0333, 0.0666, 0.2, 1.2))

    if obj.weights['vgg19']:
        obj.vgg19_loss = losses.PerceptualLoss(num_scales=args.vgg19_num_scales, use_fp16=False)

    if obj.weights['vgg19_face']:
        obj.vgg19_loss_face = losses.PerceptualLoss(num_scales=2, network='vgg_face_dag',
                                                    layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
                                                    resize=True, weights=(0.03125, 0.0625, 0.125, 0.25, 1.0), use_fp16=False)

    if obj.weights['face_resnet']:
        obj.face_resnet_loss= losses.PerceptualLoss(num_scales=1, network='face_resnet',
                                                    layers=['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12', 'relu13', 'relu14', 'relu15', 'relu16'],
                                                    resize=True, weights=(0.00, 0.00, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.6, 1, 5),
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
    data_dict['source_rotation_warp_2d'] = data_dict['source_rotation_warp'].mean(1)[..., :2]
    data_dict['source_xy_warp_resize'] = data_dict['source_xy_warp_resize'].mean(1)[..., :2]
    data_dict['target_uv_warp_resize_2d'] = data_dict['target_uv_warp_resize'].mean(1)[..., :2]
    data_dict['target_rotation_warp_2d'] = data_dict['target_rotation_warp'].mean(1)[..., :2]

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

    data_dict['source_motion_img'] = F.grid_sample(
        data_dict['source_img'],
        data_dict['source_rotation_warp_2d'],
        padding_mode='reflection')

    data_dict['target_motion_img'] = F.grid_sample(
        F.grid_sample(
            data_dict['source_motion_img'],
            data_dict['target_uv_warp_resize_2d']),
        data_dict['target_rotation_warp_2d'],
        padding_mode='reflection')


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

    data_dict['s_masked'] = data_dict['source_img'] * data_dict['source_mask']
    data_dict['sb_masked'] = data_dict['source_img'] * (1 - data_dict['source_mask'])
    data_dict['t_masked'] = data_dict['target_img'] * data_dict['target_mask']
    data_dict['target_mask_p'] = data_dict['target_mask']

    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        else:
            continue

        if 'driver' in k or 'target' in k:
            v = v.view(b, -1, *v.shape[1:])

            for i in range(min(t, 2)):
                visuals_data_dict[f'{k}_{i}'] = v[:, i]

        else:
            visuals_data_dict[k] = v

    visuals = []

    uvs_prep = lambda x: (x.permute(0, 3, 1, 2) + 1) / 2
    segs_prep = lambda x: torch.cat([x] * 3, dim=1)
    scores_prep = lambda x: (x + 1) / 2
    confs_prep = lambda x: x / x.max(dim=2, keepdims=True)[0].max(dim=3, keepdims=True)[0]

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
        ['tal', None]


    ]

    for i in range(min(t, 2)):
        visuals_list += [
            [f'target_face_img_{i}', None],
            [f'target_img_to_dis_{i}', None],
            [f'target_img_{i}', None],

            # [f'target_img_align_{i}', None],

            # [f'target_mask_{i}', None],
            [f'pred_target_img_{i}', None],
            [f'pred_mixing_img', None],
            ['rolled_mix', None],
            [f'pred_target_img_face_align_{i}', None],
            # [f'pred_target_img_face_align_retina_{i}', None],
            [f'target_img_align_orig_{i}', None],
            [f'pred_target_seg_{i}', None],
            [f'target_mask_p_{i}', None],
            # [f'target_img_align_orig_retina_{i}', None],
            # [f'pred_target_img_flip_{i}', None],
            [f'target_stickman_{i}', None],
            [f'target_landmarks_{i}', None],
            [f'target_warp_aug_{i}', None],
            ['pred_landmarks', None],

            [f'target_vgg19_conf_{i}', confs_prep],
            [f'target_vgg19_conf_flip_{i}', confs_prep]]

    # for i in range(min(t, 2)):
    #     visuals_list += [
    #         [f'target_motion_warp_2d_{i}', uvs_prep],
    #         # [f'target_rotation_warp_2d_{i}', uvs_prep],
    #         [f'target_uv_warp_resize_2d_{i}', uvs_prep],
    #         [f'target_warp_aug_{i}', None]
    #     ]

    visuals_list += [
        # ['pred_mixing_img', None],
        ['mixing_img_align', None],
        ['target_img_align_orig',None],
        ['target_img_align_orig_0', None],
        ['pred_mixing_mask', None],
        ['pred_mixing_seg', None],

        ['pred_identical_cycle', None],
        # ['rolled_mix', None],
        ['rolled_mix_align', None],
        ['pred_expression_cycle', None],
        ['pred_mixing_seg', None],
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
        self.modnet_pass = '/Vol0/user/n.drobyshev/latent-texture-avatar/models/repos/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'
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