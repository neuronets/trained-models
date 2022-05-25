import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class UNetXd(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, convs_per_block=1, block_config=(32, 32, 32, 32), padding=0, padding_mode=['circular','reflect','reflect'], 
                 positional=0, norm=False, drop=0.0, transition=False, kernel_size=3, X=3, load_path=None, skip=True, **kwargs):
        super(UNetXd, self).__init__()

        self.positional = _UNetPositional(positional, X=X) if positional > 0 else nn.Identity()
        self.padding = _UNetPadXd(padding, X=X, padding_mode=padding_mode) if padding > 0 else nn.Identity()
        self.unpadding = _UNetUnpadXd(padding, X=X) if padding > 0 else nn.Identity()
        self.block_config = list(block_config)
        self.skip = skip

        block_config = [in_channels + 2 * positional] + self.block_config
        self.features = nn.Sequential()
        for i in range(0,len(block_config) - 1):
            block = _UNetBlock(convs_per_block, block_config[i + 0], block_config[i + 1], level=i, norm=norm, drop=drop, relu=i > 0, kernel_size=kernel_size, X=X)
            self.features.add_module('convblock%d' % (i + 1), block)
            if i != len(block_config) - 1:
                pool = _Transition(block_config[i + 1], block_config[i + 1], norm=norm, drop=drop, X=X) if transition else \
                       eval('nn.MaxPool%dd' % X)(kernel_size=2, stride=2, padding=0, return_indices=True)
                self.features.add_module('maxpool%d' % (i + 1), pool)

        block_config = [out_channels] + self.block_config
        self.upsample = nn.Sequential()
        for i in reversed(range(0,len(block_config) - 1)):
            if i != len(block_config) - 1:
                pool = _TransitionTranspose(block_config[i + 1], block_config[i + 1], norm=norm, drop=drop, X=X) if transition else \
                       eval('nn.MaxUnpool%dd' % X)(kernel_size=2, stride=2, padding=0)
                self.upsample.add_module('maxunpool%d' % (i + 1), pool)
            block = _UNetTransposeBlock(convs_per_block, block_config[i + 1], block_config[i + 0], level=i, norm=norm, drop=drop, kernel_size=kernel_size, skip=self.skip, X=X)
            self.upsample.add_module('convblock%d' % (i + 1), block)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if load_path is not None:
            self.load_state_dict(torch.load(load_path))

    def forward(self, x):
        sig = [None] * len(self.block_config)
        siz = [None] * len(self.block_config)
        out = [None] * len(self.block_config)

        x = self.positional(x)
        x = self.padding(x)
        self.upsample.maxunpool1.output_padding = tuple(1 - torch.as_tensor(x.shape[2:]) % 2)

        for i in range(0,len(self.block_config) - 1):
            x = sig[i] = self.features.__getattr__('convblock%d' % (i + 1))(x)
            if i != len(self.block_config):
                out[i] = x.shape
                x, siz[i] = self.features.__getattr__('maxpool%d' % (i + 1))(x)

        for i in reversed(range(0,len(self.block_config) - 1)):
            if i != len(self.block_config):
                x = self.upsample.__getattr__('maxunpool%d' % (i + 1))(x, siz[i], output_size = out[i])
            x = self.upsample.__getattr__('convblock%d' % (i + 1))(torch.cat([x, sig[i]], 1) if self.skip else x)

        x = self.unpadding(x)

        return x

class _UNetPositional(nn.ModuleDict):
    def __init__(self, positional, modes=[0.5, 0, 0], X=2):
        super().__init__()
        self.positional = positional
        self.modes = modes
        self.dims = X

    def forward(self, img):
        if self.positional == 0:
            return img

        grids = [torch.tensor(0., device=img.device)] * 2
        for d in range(self.dims):
            grids = grids + [torch.arange(-1, 1, 2/img.shape[2 + d], device=img.device)]
        grids = torch.meshgrid(grids)

        image = [img]
        for f in range(self.positional):
            for d in range(self.dims):
                s = math.pi * (2 ** f - self.modes[d])
                image = image + [torch.cos(s*grids[2 + d])] #, torch.cos(s*grids[2 + d])

        return torch.cat(image, 1)

class _UNetPadXd(nn.ModuleDict):
    def __init__(self, padding=16, padding_mode=['circular','replicate','replicate'], X=2):
        super(_UNetPadXd, self).__init__()
        self.padding = padding
        self.padding_dims = (padding * torch.eye(X, dtype=torch.int).repeat_interleave(2,1)).tolist()
        self.padding_mode = padding_mode
        self.dims = X

    def forward(self, features):
        for d in range(self.dims):
            features = F.pad(features, pad=self.padding_dims[d], mode=self.padding_mode[d])
        
        return features

class _UNetUnpadXd(nn.ModuleDict):
    def __init__(self, padding=16, X=2):
        super(_UNetUnpadXd, self).__init__()
        self.padding = padding
        self.dims = X

    def forward(self, features):
        for d in range(self.dims):
            features = features.index_select(d+2, torch.arange(self.padding, features.shape[d+2]-self.padding, device=features.device))

        return features

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features, norm, drop, X):
        super(_Transition, self).__init__()
        self.add_module('norm', eval('nn.BatchNorm%dd' % X)(num_input_features) if norm else nn.Identity())
        self.add_module('relu', nn.ELU())
        self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
        self.add_module('conv', eval('nn.Conv%dd' % X)(num_input_features, num_output_features, groups=num_output_features,
                                                       kernel_size=3, stride=2, padding=1, bias=False if norm else True))

    def forward(self, prev_features):
        dim = tuple(1 - torch.as_tensor(prev_features.shape[2:]) % 2)
        new_features = self.drop(self.conv(self.relu(self.norm(prev_features))))

        return new_features, dim

class _TransitionTranspose(nn.Module):
    def __init__(self, num_input_features, num_output_features, norm, drop, X):
        super(_TransitionTranspose, self).__init__()
        self.add_module('norm', eval('nn.BatchNorm%dd' % X)(num_input_features) if norm else nn.Identity())
        self.add_module('relu', nn.ELU())
        self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
        self.add_module('conv', eval('nn.ConvTranspose%dd' % X)(num_input_features, num_output_features, groups=num_output_features,
                                                                kernel_size=3, stride=2, padding=1, bias=False if norm else True))

    def forward(self, prev_features, dim, **kwargs):
        self.conv.output_padding = dim
        new_features = self.conv(self.relu(self.norm(self.drop(prev_features))))

        return new_features

class _UNetLayer(nn.ModuleDict):
    def __init__(self, num_input_features, features, relu, norm, drop, kernel_size, X):
        super(_UNetLayer, self).__init__()
        self.add_module('norm', eval('nn.BatchNorm%dd' % X)(num_input_features) if norm else nn.Identity())
        self.add_module('relu', nn.ELU() if relu else nn.Identity())
        self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
        self.add_module('conv', eval('nn.Conv%dd' % X)(num_input_features, features, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False if norm else True))

    def forward(self, features):
        return self.drop(self.conv(self.relu(self.norm(features))))

class _UNetLayerTranspose(nn.ModuleDict):
    def __init__(self, num_input_features, features, relu, norm, drop, kernel_size, X):
        super(_UNetLayerTranspose, self).__init__()
        self.add_module('norm', eval('nn.BatchNorm%dd' % X)(num_input_features) if norm else nn.Identity())
        self.add_module('relu', nn.ELU())
        self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
        self.add_module('conv', eval('nn.Conv%dd' % X)(num_input_features, features, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False if norm else True))

    def forward(self, features):
        return self.conv(self.relu(self.norm(self.drop(features))))

class _UNetBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, features, level, norm, drop, kernel_size, X, relu=True, skip=False):
        super(_UNetBlock, self).__init__()
        layer = _UNetLayer(in_channels, features, relu=relu, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
        self.add_module('convlayer%d' % (0 + 1), layer)
        for i in range(1,num_layers):
            growth = 1 + (skip and i == num_layers - 1)
            layer = _UNetLayer(features, growth * features, relu=True, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
            self.add_module('convlayer%d' % (i + 1), layer)

    def forward(self, features):
        for name, layer in self.items():
            features = layer(features)
        return features

class _UNetTransposeBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, features, level, norm, drop, kernel_size, X, skip=False):
        super(_UNetTransposeBlock, self).__init__()
        for i in reversed(range(1,num_layers)):
            growth = 1 + (skip and i == num_layers - 1)
            layer = _UNetLayerTranspose(growth * in_channels, in_channels, relu=True, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
            self.add_module('convlayer%d' % (i + 1), layer)
        layer = _UNetLayerTranspose(in_channels, features, relu=True, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
        self.add_module('convlayer%d' % (0 + 1), layer)

    def forward(self, features):
        for name, layer in self.items():
            features = layer(features)
        return features

class UNet2d(UNetXd):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, X=2, **kwargs)

class UNet3d(UNetXd):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, X=3, **kwargs)

def unet2d_240(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(16,32,64,128,256), convs_per_block=1, global_skip=global_skip, **kwargs)

def unet2d_128(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(8,16,32,64,128), convs_per_block=1, global_skip=global_skip, **kwargs)

def unet2d_320(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(24,48,96,192,384), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_320_dktatlas_positional_20_1_0_0(in_channels=3, 
                                            out_channels=32, 
                                            padding=32,
                                            loadpath ='dktatlas_identity_0.000_0.000_unet2d_320_0.050_60_pos_20_1.0.0.ckpt', 
                                            **kwargs):
    #loadpath = 'dktatlas_identity_0.000_0.000_unet2d_320_0.050_60_pos_20_1.0.0.ckpt'
    model = unet2d_320(in_channels, out_channels, padding=padding, positional=20, load_path=loadpath, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()
