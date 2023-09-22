import torch
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(ite, model, optimizer, lr_scheduler, best_val_loss, ckp_file):
    torch.save({'iteration': ite,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'best_val_loss': best_val_loss}, 
                ckp_file)

def load_checkpoint(ckp_file, model=None, optimizer=None, lr_scheduler=None):
    chk_dict = torch.load(ckp_file)
    iteration, best_val_loss = chk_dict['iteration'], chk_dict['best_val_loss']
    
    if model is not None:
        model.load_state_dict(chk_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(chk_dict['optimizer_state_dict'])
    if chk_dict['lr_schedule_state_dict'] is not None and lr_scheduler is not None:
        lr_scheduler.load_state_dict(chk_dict['lr_schedule_state_dict'])
    
    return iteration, best_val_loss
            

class DeepCSRNetwork(nn.Module):
    def __init__(self, hypercol=True, out_dim=4):
        super(DeepCSRNetwork, self).__init__()
        self.hypercol = hypercol        

        if self.hypercol:
            self.encoder =  HypercolumnNetwork()   
        else:
            self.encoder = MRIEncoderNetwork()
        self.decoder = OCCNetDecoder(512, 3, 256, out_dim=out_dim)        


    def forward(self, mris, points, transforms, precomp_feature_maps=None):
        if self.hypercol:
            feats, precomp_feature_maps = self.encoder(mris, points, transforms, precomp_feature_maps)
        else:
            feats, precomp_feature_maps = self.encoder(mris, precomp_feature_maps)

        return self.decoder(points, feats), precomp_feature_maps



#### MAIN NETWORKS

class MRIEncoderNetwork(nn.Module):    
    def __init__(self):
        super(MRIEncoderNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), 
            nn.Conv3d(8, 16, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1, stride=2), nn.ReLU(),
            nn.MaxPool3d(3, padding=1, stride=2), nn.Conv3d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.MaxPool3d(3, padding=1, stride=2), Flatten(), nn.Linear(128 * 3 * 4 * 3, 512), nn.ReLU()
        )


    def forward(self, mris, precomp_features=None):
        # read in data
        if mris.ndim == 4: mris = mris.unsqueeze(1)
        assert mris.ndim == 5
        batch_size, channels, width, height, depth = mris.shape        

        # compute feature map
        if precomp_features is None:        
            glb = self.layers(mris)
            precomp_features = [glb]

        return precomp_features[0], precomp_features


class HypercolumnNetwork(nn.Module):    

    def __init__(self):
        super(HypercolumnNetwork, self).__init__()

        self.actvn = F.relu
        self.max_pool = nn.MaxPool3d(3, padding=1, stride=2)

        self.conv_in = nn.Conv3d(1, 8, 3, padding=1)
        self.conv_0 = nn.Conv3d(8, 16, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(16, 32, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        
        self.fc = nn.Linear(128 * 3 * 4 * 3, 264)
        self.point_pool = PointPooling()

    def forward(self, mris, points, transforms, precomp_feature_maps_and_strides=None):

        # read in data                
        if mris.ndim == 4: mris = mris.unsqueeze(1)        
        assert mris.ndim == 5
        batch_size, channels, width, height, depth = mris.shape
        assert batch_size == points.shape[0]
        num_points, points_dim = points.shape[1:]

        # compute feature maps
        if precomp_feature_maps_and_strides is None:
            map_1 = self.actvn(self.conv_in(mris))
            map_2 = self.actvn(self.conv_0(map_1))
            map_3 = self.actvn(self.conv_1(map_2))
            map_4 = self.actvn(self.conv_2(self.max_pool(map_3)))
            map_5 = self.actvn(self.conv_3(map_4))                
            glb = self.actvn(self.fc(self.max_pool(map_5).view(batch_size, 128 * 3 * 4 * 3)))
            precomp_feature_maps_and_strides = [(map_1, (0.5 ** 0)), (map_2, (0.5 ** 1)), (map_3, (0.5 ** 2)), (map_4, (0.5 ** 4)), (map_5, (0.5 ** 5)), (glb, None)]    
        
        # map world coordinates to mri voxels
        p_mri = torch.cat([points, torch.ones((batch_size, num_points, 1), device=0)], dim=-1)
        p_mri = torch.matmul(transforms, p_mri.transpose(2, 1)).transpose(2, 1)[:, :, :-1]

        # hypercolumn formation    
        hypercolumns = [self.point_pool(feat_map, p_mri * stride_scale) for feat_map, stride_scale in precomp_feature_maps_and_strides[:-1]]
        hypercolumns.insert(0, torch.cat([precomp_feature_maps_and_strides[-1][0].unsqueeze(dim=-1)] * num_points, dim=-1))
        hypercolumns = torch.cat(hypercolumns, dim=1)

        return hypercolumns, precomp_feature_maps_and_strides


class OCCNetDecoder(nn.Module):
    def __init__(self, feat_dim, points_dim=3, hidden_size=256, out_dim=1):
        super(OCCNetDecoder, self).__init__()
        
        self.p_dim, self.f_dim, self.out_dim = points_dim, feat_dim, out_dim

        self.fc_p = nn.Conv1d(self.p_dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(self.f_dim, hidden_size)
        self.block1 = CResnetBlockConv1d(self.f_dim, hidden_size)
        self.block2 = CResnetBlockConv1d(self.f_dim, hidden_size)
        self.block3 = CResnetBlockConv1d(self.f_dim, hidden_size)
        self.block4 = CResnetBlockConv1d(self.f_dim, hidden_size)
        
        self.bn = CBatchNorm1d(self.f_dim, hidden_size)        
        self.fc_out = nn.Conv1d(hidden_size, self.out_dim, 1)        
        self.actvn = F.relu
        
    def forward(self, points, feats):        
        points = points.transpose(1, 2)
        batch_size, D, T = points.size()
        net = self.fc_p(points)
       
        net = self.block0(net, feats)
        net = self.block1(net, feats)
        net = self.block2(net, feats)
        net = self.block3(net, feats)
        net = self.block4(net, feats)

        out = self.fc_out(self.actvn(self.bn(net, feats)))
        out = out.transpose(1, 2)
        return out
    

#### CUSTOM LAYERS

class PointPooling(nn.Module):
    """
    Local pooling operation.
    """
    def __init__(self, interpolation='bilinear'):
        super(PointPooling, self).__init__()
        self.interp_mode = interpolation

    def forward(self, x, v):        
        batch, num_points, num_feats = x.size(0), v.size(1), x.size(1)
        norm_v = torch.tensor([[[x.size(2)-1, x.size(3)-1, x.size(4)-1]]], device='cuda').float()
        norm_v = 2 * (v/norm_v) - 1
        grid = norm_v.unsqueeze(dim=-2).unsqueeze(dim=-2).flip(dims=(-1,))

        out = F.grid_sample(x, grid, mode=self.interp_mode, padding_mode='border', align_corners=True)
        out = out.squeeze().view(batch, num_feats, num_points)
        return out


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None, norm_method='batch_norm'):
        super(CResnetBlockConv1d, self).__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in, self.size_h, self.size_out = size_in, size_h, size_out
                
        # Submodules        
        self.bn_0 = CBatchNorm1d(
            c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(
            c_dim, size_h, norm_method=norm_method)
    
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
    
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.   
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super(CBatchNorm1d, self).__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class Debug(nn.Module):
    def __init__(self, func):
        super(Debug, self).__init__()
        self.func = func

    def forward(self, x):
        self.func(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
