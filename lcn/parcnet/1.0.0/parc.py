import os
import glob
import math
import random
import torch
import torch.nn as nn
import numpy as np
# import freesurfer as fs
import nibabel as nib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

class PARC(VisionDataset):
    def __init__(
            self,
            root: str = '/autofs/space/pac_001/tiamat_copy/3/users/subjects/aparc_atlas',
            split: range = None,
            subset: str = 'DKTatlas',
            labels = 'labels.DKT31.manual.2.annot',
            inputs = ['inflated.H', 'sulc', 'curv'],
            hemisphere: str = 'rh',
            transforms: Optional[Callable] = None,
            in_channels: int = 1,
            num_classes: int = 1,
            out_shape = [32, 256, 512],
            mode = 'image',
            multiplier = 1,
            stride: int = 1,
            labeled: bool = True,
            **kwargs
    ):
        super().__init__(root)

        self.labeled = labeled
        self.stride = stride
        self.inputs = inputs
        self.labels = labels
        self.split = split #verify_str_arg(split, 'split', ['train', 'valid'])
        self.transforms = transforms
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mode = verify_str_arg(mode, 'mode', ['image', 'surf'])
        self.multiplier = multiplier

        self.hemisphere = verify_str_arg(hemisphere, 'hemisphere', ['rh', 'lh'])
        self.out_shape = out_shape

        self.sphere_file =  os.path.join('surf', '%s.sphere.reg' % (self.hemisphere))
        self.expand_file =  os.path.join('surf', '%s.inflated'   % (self.hemisphere))
        self.parcel_file =  os.path.join('label','%s.%s' % (self.hemisphere, labels))
        self.signal_file = [os.path.join('surf', '%s.%s' % (self.hemisphere, signal)) for signal in inputs]

        fullpath = os.path.join(root, subset)
        # with open(os.path.join(fullpath, '%s_%d.txt') % (split, seed)) as subjfile:
        #     self.subjects = subjfile.read().splitlines()
        self.subjects = sorted([p for p in os.listdir(fullpath) if len(glob.glob(os.path.join(fullpath, p, self.signal_file[0] + '*'))) > 0])
        self.subjects = [self.subjects[i] for i in self.split] if self.split != None else self.subjects

        self.spheres =  [os.path.join(fullpath, subject, self.sphere_file) for subject in self.subjects]
        self.expands =  [os.path.join(fullpath, subject, self.expand_file) for subject in self.subjects]
        self.parcels =  [os.path.join(fullpath, subject, self.parcel_file) for subject in self.subjects]
        self.signals = [[os.path.join(fullpath, subject, signal) for signal in self.signal_file] for subject in self.subjects]


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label, index) where label is the image segmentation.
        """

        index = index % len(self.signals)

        # if self.mode == 'surf':
        #     image = torch.tensor([fs.Surface.read(self.spheres[index]).parameterize(fs.Overlay.read(signal).data) for signal in self.signals[index]])
        #     label = torch.tensor([fs.Surface.read(self.spheres[index]).parameterize(fs.Overlay.read(self.parcels[index]).data, interp='nearest')]) if self.labeled \
        #         else torch.zeros([1] + list(image.shape[1:]))

        if self.mode == 'image':
            image = torch.tensor([nib.load(signal + '.mgz').get_fdata()[:,:,0] for signal in self.signals[index]], dtype=torch.float32).permute(0,2,1)
            label = torch.tensor([nib.load(self.parcels[index] + '.mgz').get_fdata()], dtype=torch.float32).permute(0,2,1) if self.labeled else torch.zeros([1] + list(image.shape[1:]))

        if self.transforms != None:
            image, label = self.transforms(image, label)

        return image, label, index

    def save_output(self, root, outputs, indices):
        for i in range(0, len(outputs)):
            # filename = self.subjects[i] '%05d' % (indices[i])
            # if self.mode == 'surf':
            #     parcel = fs.Overlay.read(self.parcels[indices[i]])
            #     parcel.write(os.path.join(root,filename + '.label.annot'))
            #     sphere = fs.Surface.read(self.spheres[indices[i]])
            #     parcel = fs.Overlay(sphere.sample_parameterization(outputs[i].cpu().numpy(), interp='nearest'), lut=parcel.lut)
            #     parcel.write(os.path.join(root,filename + '.image.annot'))

            if self.mode == 'image':
                # parcel = fs.Image.read(self.parcels[indices[i]] + '.mgz')
                # parcel.write(os.path.join(root,filename + '.label.mgz'))
                parcel = nib.MGHImage(outputs[i].permute(0,2,1).short().cpu().numpy(), np.eye(4))#  fs.Image(outputs[i].permute(0,2,1).cpu())
                os.makedirs(os.path.join(root, self.subjects[indices[i]]), exist_ok=True)
                nib.save(parcel, os.path.join(root, self.subjects[indices[i]], 'parc.mgz')) #parcel.write(os.path.join(root,filename + '.image.mgz'))

    def __len__(self) -> int:
        return int(len(self.signals) * self.multiplier)

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.in_channels

    def __numclass__(self) -> int:
        return self.num_classes

    def __weights__(self) -> torch.Tensor:
        # weight_s = (torch.sin(math.pi * (1/(256-1)) * torch.arange(0, 256))**2).reshape(-1, 1).repeat(1, 512).reshape(-1,256,512)

        return 1 #weight_c #1 #weight_s #*weight_c
