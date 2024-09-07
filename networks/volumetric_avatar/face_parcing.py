import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import sys
import os


class FaceParsing(object):
    def __init__(self,
                 mask_type,
                 device="cuda",
                 project_dir = '/fsx/nikitadrobyshev/EmoPortraits',):
        super(FaceParsing, self).__init__()


        path_to_face_parsing = f'{project_dir}/repos/face_par_off'

        
        sys.path.append(path_to_face_parsing)
        sys.path.append(project_dir)
        from repos.face_par_off.model import BiSeNet

        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes).to(device)
        save_pth = os.path.join(f'{path_to_face_parsing}/res/cp/79999_iter.pth')
        self.net.load_state_dict(torch.load(save_pth, map_location='cpu'))
        self.net.eval()

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)

        self.mask_labels = []

        if mask_type is None:
            self.mask_labels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 7, 8, 9, 14, 17, 18] # face, ears, hair, hat, hair, neck
            self.face_labels = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 7, 8, 9, 17, 18] # face, ears, hair, hat, hair, neck
            self.body_labels = [18] # neck, hat, hair
            self.cloth_labels = [16,] 
        else:
            if 'face' in mask_type:
                self.mask_labels += [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
            if 'ears' in mask_type:
                self.mask_labels += [7, 8, 9]
            if 'neck' in mask_type:
                self.mask_labels += [14, 15]
            if 'hair' in mask_type:
                self.mask_labels += [17]
            if 'hat' in mask_type:
                self.mask_labels += [18]
            if 'cloth' in mask_type:
                self.mask_labels += [16]


    @torch.no_grad()
    def forward(self, x):
        h, w = x.shape[2:]
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        x = F.interpolate(x, size=(512, 512), mode='bilinear')
        y = self.net(x)[0]
        y = F.interpolate(y, size=(h, w), mode='bilinear')

        labels = y.argmax(1, keepdim=True)

        mask = torch.zeros_like(labels)
        for i in self.mask_labels:
            mask += labels == i

        mask_body = torch.zeros_like(labels)
        for i in self.body_labels:
            mask_body += labels == i

        mask_cloth = torch.zeros_like(labels)
        for i in self.cloth_labels:
            mask_cloth += labels == i

        face_body = torch.zeros_like(labels)
        for i in self.face_labels:
            face_body += labels == i

        return mask, face_body, mask_body, mask_cloth