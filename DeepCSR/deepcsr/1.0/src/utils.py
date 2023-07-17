import time
import torch
import numpy as np
import nibabel as nib


class TicToc:
    """
    TicToc class for time pieces of code.
    """

    def __init__(self):
        self._TIC_TIME = {}
        self._TOC_TIME = {}

    def tic(self, tag=None):
        """
        Timer start function
        :param tag: Label to save time
        :return: current time
        """
        if tag is None:
            tag = 'default'
        self._TIC_TIME[tag] = time.time()
        return self._TIC_TIME[tag]

    def toc(self, tag=None):
        """
        Timer ending function
        :param tag: Label to the saved time
        :param fmt: if True, formats time in H:M:S, if False just seconds.
        :return: elapsed time
        """
        if tag is None:
            tag = 'default'
        self._TOC_TIME[tag] = time.time()

        if tag in self._TIC_TIME:
            d = (self._TOC_TIME[tag] - self._TIC_TIME[tag])
            return d
        else:
            print("No tic() start time available for tag {}.".format(tag))

    # Timer as python context manager
    def __enter__(self):
        self.tic('CONTEXT')

    def __exit__(self, type, value, traceback):
        self.toc('CONTEXT')



def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def save_nib_image(path, voxel_grid, affine=np.eye(4), header=None):
    nib_img = nib.Nifti1Image(voxel_grid, affine, header)
    nib.save(nib_img, path)