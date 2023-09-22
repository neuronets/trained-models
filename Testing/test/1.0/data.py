import logging, os, random
from torch.utils import data
import numpy as np
import torch
import trimesh, nibabel


logger = logging.getLogger(__name__)


class CSRDataset(data.Dataset):

    def __init__(self, dataset_path, split_name, split_path, data_fields, surfaces, shuffle=False, field_transform=None, data_transform=None):
        super(CSRDataset, self).__init__()

        self.dataset_path = dataset_path
        self.split_name = split_name
        self.split_path = split_path
        self.data_fields = data_fields
        self.surfaces = surfaces
        self.field_transform = field_transform        
        self.data_transform = data_transform
        assert all( field in ['mri', 'points', 'pointcloud', 'transform', 'mesh'] for field in self.data_fields)
        assert all(surf in ['lh_pial', 'lh_white', 'rh_pial', 'rh_white'] for surf in self.surfaces)

        # read in data
        self.subjects = []
        with open(os.path.join(self.dataset_path, split_path), 'r') as split_file:
            for subject in split_file.readlines():
                subject = subject.strip()
                self.subjects.append(subject)
        if shuffle:
            random.shuffle(self.subjects)

    def __len__(self):               
        return len(self.subjects)

    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        data = {}
        
        # load data fields and apply transforms
        for field in self.data_fields:
            if field == 'mri':            
                _, mri_vox, mri_affine = mri_reader(os.path.join(self.dataset_path, subject, 'mri.nii.gz'))                
                data['mri_vox'], data['mri_affine'] = mri_vox, mri_affine                 

            elif field == 'points':
                points, occs, dists = points_reader(os.path.join(self.dataset_path, subject, 'points.{}.npz'.format(self.split_name)), self.surfaces)
                data['pts_loc'], data['pts_occ'], data['pts_dist'] = points, occs, dists
                
            elif field == 'pointcloud':
                points, normals, surf_ids = pointcloud_reader(os.path.join(self.dataset_path, subject, 'pointcloud.{}.npz'.format(self.split_name)), self.surfaces)
                data['pcl_loc'], data['pcl_normal'], data['pcl_surf'] = points, normals, surf_ids
                
            elif field == 'transform':
                transform = transform_reader(os.path.join(self.dataset_path, subject, 'transform_affine.txt'))
                data['transform'] = transform

            elif field == 'mesh':
                for surf in self.surfaces:
                    mesh_vertices, mesh_faces = mesh_reader(os.path.join(self.dataset_path, subject, '{}.stl'.format(surf)))
                    data['{}_verts'.format(surf)], data['{}_faces'.format(surf)] = mesh_vertices, mesh_faces

            # apply field specific transform
            if field in self.field_transform:
                data = self.field_transform[field](data)                

        # apply data specific transform
        if self.data_transform is not None:
            data = self.data_transform(data)
        
        return data


def collate_CSRData_fn(batch_list):

    # simply stack for points, mri, pointcloud, and transform
    batch_data = {key: [] for key in batch_list[0].keys()}    
    for key in ['mri_vox', 'mri_affine', 'transform', 'pts_loc', 'pts_occ', 'pts_dist', 'pts_isrpr', 'pcl_loc', 'pcl_normal', 'pcl_surf']:
        if key in batch_data:
            batch_data[key] = np.stack([data[key] for data in batch_list], axis=0)

    
    # meshes in packed format: concatenated vertices and faces and the count of each i a Bx2 tensor
    surfaces = [key.replace('_verts', '') for key in batch_data if key.endswith('_verts')]
    if len(surfaces) > 0:        
        for surf in surfaces:
            vertices_key, faces_key, lenghts_key = '{}_verts'.format(surf), '{}_faces'.format(surf), '{}_lenghts'.format(surf)                         
            vertices_list, faces_list = [data[vertices_key] for data in batch_list], [data[faces_key] for data in batch_list]
            batch_data[vertices_key], batch_data[faces_key], batch_data[lenghts_key] = pack_meshes(vertices_list, faces_list)

    # make everything tensor
    for key in batch_data:        
        batch_data[key] = torch.from_numpy(batch_data[key])

    return batch_data


def pack_meshes(vertices_list, faces_list):
    packed_vertices, packed_faces, packed_lenghts, vertices_cumsum = [], [], [], 0.0
    assert len(vertices_list) == len(faces_list)
    for verts, faces in zip(vertices_list, faces_list):
        packed_vertices.append(verts)
        packed_faces.append(faces +  vertices_cumsum)
        packed_lenghts.append([verts.shape[0], faces.shape[0]])
        vertices_cumsum += verts.shape[0]    
    return np.concatenate(packed_vertices, axis=0), np.concatenate(packed_faces, axis=0), np.array(packed_lenghts)


# READERS

def mri_reader(path):
    nib_mri = nibabel.load(path)
    mri_header, mri_vox, mri_affine = nib_mri.header, nib_mri.get_fdata().astype(np.float32), nib_mri.affine.astype(np.float32)
    # nibabel voxel to world cordinates affine
    return mri_header, mri_vox, mri_affine

def points_reader(path, surfaces):
    arrays = np.load(path)    
    surfnames = arrays['surfnames'].tolist()
    surf_select = arrays['surf2idx'][[surfnames.index(surf) for surf in surfaces]]

    points, occupancies, distances = arrays['points'], arrays['occupancies'], arrays['distances']
    # compressed format -> unpackbits
    if occupancies.shape[0] != points.shape[0]:
        occupancies = np.unpackbits(occupancies).reshape(points.shape[0], -1)
    
    occupancies = occupancies[:, surf_select].astype(np.float32)
    points = points.astype(np.float32)
    distances = distances[:, surf_select].astype(np.float32)

    return points, occupancies, distances

def pointcloud_reader(path, surfaces):
    arrays = np.load(path)
    surfnames = arrays['surfnames'].tolist()
    surf_select = np.ones_like(arrays['ids']) * -1
    for s_idx, surf in enumerate(surfaces):
        surf_select[arrays['ids'] == surfnames.index(surf)] = s_idx
    surf_mask = surf_select >= 0        
    return arrays['points'][surf_mask].astype(np.float32), arrays['normals'][surf_mask].astype(np.float32), surf_select[surf_mask].astype(np.float32)

def mesh_reader(path):
    mesh = trimesh.load(path)
    return np.array(mesh.vertices).astype(np.float32), np.array(mesh.faces).astype(np.int32)

def transform_reader(path):
    transform = np.loadtxt(path)
    return transform


# TRANSFORMS

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        super(SubsamplePointcloud, self).__init__()
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data['pcl_loc']
        normals = data['pcl_normal']
        surfs = data['pcl_surf']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out['pcl_loc'] = points[indices, :]
        data_out['pcl_normal'] = normals[indices, :]
        data_out['pcl_surf'] = surfs[indices]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the data points.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        super(SubsamplePoints, self).__init__()
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data['pts_loc']
        occ = data['pts_occ']
        distances = data['pts_dist']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                'pts_loc': points[idx, :],
                'pts_occ':  occ[idx],
                'pts_dist' : distances[idx]
            })
            
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            distances0 = distances[~occ_binary][idx0]
            distances1 = distances[occ_binary][idx1]
            distances = np.concatenate([distances0, distances1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                'pts_loc': points,
                'pts_occ': occ,
                'pts_volume': volume,
                'pts_dist': distances
            })

        return data_out


class NormalizeMRIVoxels(object):
    '''
    Normalize input voxel data
    '''
    def __init__(self, norm_type='mean_std', **kwargs):
        """
        normalize transform for inputs
        :param norm_type: type of normalization mean-std or min-max
        :param kwargs: parameters for normalization
        """
        super(NormalizeMRIVoxels, self).__init__()

        self.norm_type = norm_type
        self.args_dict = kwargs

    def __call__(self, data):
        """
        Apply normalization to data
        :param data: data point
        :return: data points with normalized inputs
        """
        if self.norm_type == 'mean_std':
            mean = float(self.args_dict.get('mean', data['mri_vox'].mean()))
            std = float(self.args_dict.get('std', data['mri_vox'].std()))
            data['mri_vox'] = (data['mri_vox'] - mean) / std

        elif self.norm_type == 'min_max':
            min, max = float(data['mri_vox'].min()), float(data['mri_vox'].max())
            scale = float(self.args_dict.get('scale', 1.0))
            data['mri_vox'] = ((data['mri_vox'] - min) / (max - min)) * scale

        else:
            raise ValueError('{} normalization is not supported'.format(self.norm_type))

        return data


class InvertAffine(object):
    '''
    Invert Affine
    '''
    def __init__(self, affine_data_key):
        super(InvertAffine, self).__init__()

        self.affine_data_key = affine_data_key        

    def __call__(self, data):
        data[self.affine_data_key] = np.linalg.inv(data[self.affine_data_key]).astype(np.float32)
        return data


class PointsToImplicitSurface(object):    
    def __init__(self, implicit_rpr):
        super(PointsToImplicitSurface, self).__init__()
        assert implicit_rpr in ['sdf', 'occ']
        self.impl_rpr = implicit_rpr

    def __call__(self, data):
        occ, dist = data['pts_occ'], data['pts_dist']

        if self.impl_rpr == 'sdf':
            sdf = dist * (occ + ((1.0 - occ) * (-1.)))
            data['pts_isrpr'] = sdf 
        else:
            data['pts_isrpr'] = data['pts_occ']
        
        data.pop('pts_occ');data.pop('pts_dist')
        return data
