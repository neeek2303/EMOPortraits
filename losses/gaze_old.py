# import torch
# import torch.nn.functional as F
# from pathlib import Path
# from torch import nn
# import cv2
# import numpy as np
#
# from rt_gene import RtGeneEstimator
# from runners import utils as rn_utils
#
# class LossWrapper(nn.Module):
#     @staticmethod
#     def get_args(parser):
#         parser.add('--gaze_weight', type=float, default=20.0)
#         parser.add('--gaze_loss_type', choices=['mse', 'l1'], default='l1')
#         parser.add('--gaze_loss_pred_img_names', type=str, default='pred_target_imgs')
#         parser.add('--gaze_loss_target_img_name', type=str, default='target_imgs')
#         parser.add('--gaze_loss_calc_landmarks', default='True', type=rn_utils.str2bool, choices=[True, False])
#         parser.add('--gaze_loss_names', type=str, default='gaze')
#
#         parser.add('--rt_gene_model_nets_path', type=Path,
#                    default=Path('/group-volume/orc_srr/violet/d.nikulin/models/rt_gene/'),
#                    help='Path to directory with RT-GENE model checkpoints')
#         parser.add('--rt_gene_criterion_model_ids', type=int, nargs='+', default=(3,),
#                    help='List of models to include in the RT-GENE ensemble. Can be any subset of (1,2,3,4)')
#         parser.add('--rt_gene_layer_indices', type=int, nargs='+', default=(1, 6, 11, 18, 25),
#                    help='List of indices of conv layers to compute eye embeddings from')
#         parser.add('--rt_gene_layer_weights', type=float, nargs='+', default=(1.0, 1e-1, 4e-3, 2e-6, 1e-8),
#                    help='Weights for loss computed on eye embeddings from each of the layers')
#
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#
#         rt_gene_model_nets_path = '/group-volume/orc_srr/violet/d.nikulin/models/rt_gene/'
#         rt_gene_criterion_model_ids = 3
#         rt_gene_layer_indices = (1, 6, 11, 18, 25)
#         rt_gene_layer_weights = (1.0, 1e-1, 4e-3, 2e-6, 1e-8)
#         gaze_weight = 20
#         gaze_loss_type = 'l1'
#         gaze_loss_pred_img_names = 'pred_target_imgs'
#         gaze_loss_target_img_name = 'target_imgs'
#         gaze_loss_calc_landmarks = True
#         self.rt_gene_estimator = RtGeneEstimator(
#             device='cuda',
#             model_nets_path=args.rt_gene_model_nets_path,
#             gaze_model_ids=args.rt_gene_criterion_model_ids,
#             align_face=args.gaze_loss_calc_landmarks,
#             replace_maxpool_with_avgpool=True,
#         )
#
#         self.weight = args.gaze_weight
#         self.loss_type = args.gaze_loss_type
#         self.layer_indices = args.rt_gene_layer_indices
#         self.layer_weights = args.rt_gene_layer_weights
#
#         self.target_img_name = args.gaze_loss_target_img_name
#         self.pred_img_names = rn_utils.parse_str_to_list(args.gaze_loss_pred_img_names, sep=',')
#         self.loss_names = rn_utils.parse_str_to_list(args.gaze_loss_names, sep=',')
#
#     def forward(self, data_dict, losses_dict):
#         real_rgb = data_dict[self.target_img_name]  # (B, T, 3, 256, 256)
#
#         b, t = real_rgb.shape[:2]
#         real_rgb = real_rgb.view(b * t, *real_rgb.shape[2:])
#         real_rgb = (real_rgb + 1.0) / 2.0
#         if real_rgb.shape[2] < 256:
#             real_rgb = F.interpolate(real_rgb, size=256, mode='bicubic')
#
#         assert not real_rgb.requires_grad
#
#         # Extract landmarks if they were pre-calculated
#         real_facebbox_names = self.target_img_name.replace('imgs', 'bboxes')
#         real_landmark_names = self.target_img_name.replace('imgs', 'landmarks')
#
#         if real_facebbox_names in data_dict.keys() and real_landmark_names in data_dict.keys():
#             faceboxes = data_dict[real_facebbox_names]
#             landmarks = data_dict[real_landmark_names]
#
#         else:
#             faceboxes = landmarks = None
#
#         with torch.no_grad():
#             real_subjects = self.rt_gene_estimator.get_eye_embeddings(real_rgb, self.layer_indices, faceboxes,
#                                                                       landmarks)
#
#         indices = []  # indices of filtered subjects
#         real_subjects_ = []  # filtered subjects without None
#
#         for i, subject in enumerate(real_subjects):
#             if subject is not None and subject.eye_embeddings is not None:
#                 indices += [i]
#                 real_subjects_ += [subject]
#
#         faceboxes = [subject.box for subject in real_subjects_]
#         landmarks = [subject.landmarks for subject in real_subjects_]
#
#         if not len(landmarks):
#             return data_dict, losses_dict
#
#         # Calculate eye masks
#         eye_masks = torch.zeros_like(real_rgb)
#         eye_masks[indices] = self.draw_eye_masks([subject.landmarks for subject in real_subjects_],
#                                                  max(self.args.image_size, 256)).to(real_rgb.device)
#
#         for i, pred_img_name in enumerate(self.pred_img_names):
#             fake_rgb = data_dict[pred_img_name]  # (B, T, 3, 256, 256)
#             fake_rgb = fake_rgb.view(b * t, *fake_rgb.shape[2:])
#             fake_rgb = (fake_rgb + 1.0) / 2.0
#             if fake_rgb.shape[2] < 256:
#                 fake_rgb = F.interpolate(fake_rgb, size=256, mode='bicubic')
#
#             # Get gradients w.r.t. this tensor
#             fake_rgb_ = fake_rgb.clone().detach()[indices].requires_grad_()
#
#             fake_subjects = self.rt_gene_estimator.get_eye_embeddings(fake_rgb_, self.layer_indices, faceboxes,
#                                                                       landmarks)
#
#             embeddings = [
#                 (fake_subject.eye_embeddings, real_subject.eye_embeddings)
#                 for (fake_subject, real_subject)
#                 in zip(fake_subjects, real_subjects_)
#                 if fake_subject is not None and
#                    fake_subject.eye_embeddings is not None
#             ]
#
#             if not embeddings:
#                 continue
#
#             loss_func = {
#                 'mse': F.mse_loss,
#                 'l1': F.l1_loss,
#             }[self.loss_type]
#
#             loss_terms = []
#             num_samples = len(embeddings)
#             for fake_embs, real_embs in embeddings:
#                 # (fake_embs, real_embs) are embeddings for one of the input images
#                 for layer_weight, fake_emb, real_emb in zip(self.layer_weights, fake_embs, real_embs):
#                     # (fake_emb, real_emb) are single-layer embeddings for one of the input images
#                     assert not real_emb.requires_grad
#                     loss_terms.append((layer_weight / num_samples) * loss_func(fake_emb, real_emb))
#
#             loss = self.weight * torch.stack(loss_terms, dim=0).sum(0)
#             loss.backward()
#
#             fake_rgb_grad = torch.zeros_like(fake_rgb)
#             fake_rgb_grad[indices] = fake_rgb_.grad
#
#             losses_dict[f'{self.loss_names[i]}'] = loss.detach()
#             losses_dict[f'_{self.loss_names[i]}'] = (fake_rgb * fake_rgb_grad * eye_masks).sum()
#
#             # Normalize masked grads into [-1, 1] for visualization
#             grad_min = fake_rgb_grad.view(b * t, 3, -1).min(2)[0][..., None, None]
#             grad_max = fake_rgb_grad.view(b * t, 3, -1).max(2)[0][..., None, None]
#             fake_rgb_grad_ = (fake_rgb_grad - grad_min) / (grad_max - grad_min)
#             fake_rgb_grad_ = fake_rgb_grad_ * eye_masks
#
#             data_dict[f'{pred_img_name}_eye_grad'] = (fake_rgb_grad_ - 0.5) * 2
#
#         return data_dict, losses_dict
#
#     @staticmethod
#     def draw_eye_masks(poses, image_size):
#         ### Define drawing options ###
#         edges_parts = [list(range(36, 42)), list(range(42, 48))]
#
#         mask_kernel = np.ones((5, 5), np.uint8)
#
#         ### Start drawing ###
#         facemasks = []
#
#         for xy in poses:
#             xy = xy[None, :, None].astype(np.int32)
#
#             facemask = np.zeros((image_size, image_size, 3), np.uint8)
#
#             for edges in edges_parts:
#                 facemask = cv2.fillConvexPoly(facemask, xy[0, edges], (255, 255, 255))
#
#             facemask = cv2.dilate(facemask, mask_kernel, iterations=1)
#             facemask = cv2.blur(facemask, mask_kernel.shape)
#             facemask = torch.FloatTensor(facemask[:, :, [0]].transpose(2, 0, 1)) / 255.
#             facemasks.append(facemask)
#
#         facemasks = torch.stack(facemasks)
#
#         return facemasks