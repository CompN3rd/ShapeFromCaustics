import sys
import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from functools import wraps
from itertools import product

__authors__ = "Marc Kassubeck"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Marc Kassubeck"
__email__ = "kassubeck@cg.cs.tu-bs.de"
__status__ = "Development"


def dot_product(a, b, dim=-1, keepdim=False, normal=False):
    ret = (a * b).sum(dim=dim, keepdim=keepdim)
    return ret.clamp(min=-1, max=1) if normal else ret


# TODO: replace by https://pytorch.org/docs/stable/nn.functional.html?highlight=normalize#torch.nn.functional.normalize
def normalize_tensor(tensor, p=2, dim=-1):
    tensor_norm = tensor.norm(p=p, dim=dim, keepdim=True)
    mask = tensor_norm.gt(0).expand_as(tensor)
    ret = th.zeros_like(tensor)
    ret[mask] = (tensor / tensor_norm)[mask]
    return ret


def compute_orthonormal_basis(normal, dim='dim', eps=1e-5):
    # code adapted from Frisvad 2012: Building an Orthonormal Basis from a 3D Unit Vector Without Normalization
    nx = normal.select(dim, 0).rename(None)
    ny = normal.select(dim, 1).rename(None)
    nz = normal.select(dim, 2).rename(None)

    b1 = th.zeros_like(normal).rename(None)
    b2 = th.zeros_like(normal).rename(None)

    singulariy_mask = (nz < (-1 + eps))
    b1[1, singulariy_mask] = -1
    b2[0, singulariy_mask] = -1

    not_singularity_mask = th.logical_not(singulariy_mask)
    a = 1 / (1 + nz[not_singularity_mask])
    b = -nx[not_singularity_mask] * ny[not_singularity_mask] * a

    b1[:, not_singularity_mask] = th.stack((1 - a * nx[not_singularity_mask]**2, b, -nx[not_singularity_mask]))
    b2[:, not_singularity_mask] = th.stack((b, 1 - a * ny[not_singularity_mask]**2, -ny[not_singularity_mask]))

    return b1.refine_names(*normal.names), b2.refine_names(*normal.names)


def gram_schmidt(normal, tangent, bitangent, p=2, dim=-1):
    ret_normal = normalize_tensor(normal, p=p, dim=dim)
    ret_tangent = normalize_tensor(tangent - (ret_normal * tangent).sum(dim, keepdim=True) * tangent, p=p, dim=dim)
    ret_bitangent = bitangent - (ret_normal * bitangent).sum(dim, keepdim=True) * bitangent
    ret_bitangent = normalize_tensor(ret_bitangent - (ret_tangent * ret_bitangent).sum(dim, keepdim=True) * ret_bitangent, p=p, dim=dim)

    return ret_normal, ret_tangent, ret_bitangent


def barycentric_interpolate(buffer, tri_index, barys):
    # buffer.shape = T x v x c
    # tri_index.shape = M
    # buffer[ti].shape = M x v x c
    # bary_interpolator.shape = M x v
    bary_interpolator = th.zeros(tri_index.size(0), buffer.size(-2), dtype=buffer.dtype, device=buffer.device)
    bary_interpolator[:, 0] = 1 - barys.sum(1)
    bary_interpolator[:, 1:] = barys

    return (bary_interpolator.unsqueeze(-1) * buffer[tri_index]).sum(-2)


def slerp(p0, p1, t, dim=-1, normalize=False):
    if normalize:
        p0 = normalize_tensor(p0, dim=dim)
        p1 = normalize_tensor(p1, dim=dim)
    omega = th.acos(dot_product(p0, p1, dim=dim, keepdim=True, normal=True))
    return (th.sin((1 - t) * omega) * p0 + th.sin(t * omega) * p1) / th.sin(omega)


def barycentric_slerp(corners, barys, normalize=False):
    # corners.shape = N x 3 x c
    # barys.shape = N x 2
    a = slerp(corners[:, 0, :], corners[:, 2, :], barys[:, 1].unsqueeze(-1), normalize=normalize)
    b = slerp(corners[:, 1, :], corners[:, 2, :], barys[:, 1].unsqueeze(-1), normalize=normalize)
    return slerp(a, b, barys[:, 0].unsqueeze(-1))


def subtended_angle(positions, vertices, return_uncumulated=False):
    diffs = vertices.align_to('triangle', 'vertex', 'sample', 'dim') - positions.align_to('triangle', 'vertex', 'sample', 'dim')
    diffs_norm = th.norm(diffs.rename(None), p=2, dim=-1).refine_names('triangle', 'vertex', 'sample')

    numerators = dot_product(diffs.select('vertex', 0), th.cross(diffs.select('vertex', 2).rename(None), diffs.select('vertex', 1).rename(None), dim=-1).refine_names('triangle', 'sample', 'dim'), dim='dim')
    inner_dots = th.stack((dot_product(diffs.select('vertex', 1), diffs.select('vertex', 2), dim='dim').rename(None),
                           dot_product(diffs.select('vertex', 0), diffs.select('vertex', 2), dim='dim').rename(None),
                           dot_product(diffs.select('vertex', 0), diffs.select('vertex', 1), dim='dim').rename(None)),
                          dim=1).refine_names('triangle', 'vertex', 'sample')
    denominators = diffs_norm.prod(dim='vertex') + dot_product(diffs_norm, inner_dots, dim='vertex')

    numerators.rename_(None)
    denominators.rename_(None)
    ret = th.zeros_like(numerators)
    mask = (th.logical_not(numerators.eq(0) & denominators.eq(0)))
    ret[mask] = 2 * th.atan2(numerators[mask], denominators[mask])

    ret.rename_('triangle', 'sample')
    return ret if return_uncumulated else F.relu(ret, inplace=True).sum(dim='triangle')


def height_field_to_mesh(coords: th.Tensor, height_field: th.Tensor, height_field_reference: th.Tensor):
    points = th.cat((coords, height_field.align_to('dim', ...)), dim='dim').align_to('height', 'width', 'dim').rename(None).view(1, -1, 3)
    indices = th.arange(0, points.size(1), dtype=th.int).view(coords.size('height'), coords.size('width'))
    faces = th.zeros(1, 2 * (coords.size('width') - 1) * (coords.size('height') - 1), 3, dtype=th.int, device=coords.device)  # .refine_names('batch', 'vertex', 'dim')

    error = th.abs(height_field_reference.rename(None) - height_field.rename(None)).flatten()
    error /= error.max()

    colors = (th.tensor([[0, 0, 255]], dtype=error.dtype, device=error.device) * (1 - error).unsqueeze(1) + th.tensor([[255, 0, 0]], dtype=error.dtype, device=error.device) * error.unsqueeze(1)).unsqueeze(0)

    # clockwise orientation
    # one quad
    # *----*
    # |   /|
    # |  / |
    # | /  |
    # *----*

    faces[0, ::2, 0] = indices[:-1, :-1].flatten()
    faces[0, ::2, 1] = indices[:-1, 1:].flatten()
    faces[0, ::2, 2] = indices[1:, :-1].flatten()

    faces[0, 1::2, 0] = indices[:-1, 1:].flatten()
    faces[0, 1::2, 1] = indices[1:, 1:].flatten()
    faces[0, 1::2, 2] = indices[1:, :-1].flatten()

    return points, colors, faces


def tensorboard_logger(print_step=-1, print_memory=False):
    tensorboard_logger.writer: SummaryWriter
    tensorboard_logger.global_step: int
    tensorboard_logger.key: str

    # for each function decorated by this: how many steps need to pass for it to print it's stuff
    tensorboard_logger._print_steps = {}

    def log_tensor(key: str, tensor_callback):
        calling_func = sys._getframe(1).f_code.co_name
        if tensorboard_logger._print_steps[calling_func] > 0 and (tensorboard_logger.global_step == 1 or tensorboard_logger.global_step % tensorboard_logger._print_steps[calling_func] == 0):
            v = tensor_callback().detach()
            vmin = v.min().item()
            vmax = v.max().item()

            arr = v.cpu().numpy()
            fig = plt.figure()
            if arr.ndim == 2:
                plt.imshow(arr, vmin=vmin, vmax=vmax)
                plt.colorbar()
                tensorboard_logger.writer.add_figure("{}/{}".format(tensorboard_logger.key, key), fig, global_step=tensorboard_logger.global_step)
            elif arr.ndim == 3:
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(1, arr.shape[0]),
                                 axes_pad=0.15,
                                 share_all=True,
                                 cbar_location='right',
                                 cbar_mode='single',
                                 cbar_size='7%',
                                 cbar_pad=0.15,
                                 )
                for i, ax in enumerate(grid):
                    im = ax.imshow(arr[i], vmin=vmin, vmax=vmax)

                ax.cax.colorbar(im)
                ax.cax.toggle_label(True)
                tensorboard_logger.writer.add_figure("{}/{}".format(tensorboard_logger.key, key), fig, global_step=tensorboard_logger.global_step)

                # also add as image, if it has 3 channels
                if arr.shape[0] == 3:
                    tensorboard_logger.writer.add_image("{}/{}_RGB".format(tensorboard_logger.key, key), (arr - vmin) / (vmax - vmin), global_step=tensorboard_logger.global_step, dataformats='CHW')
            elif arr.ndim == 4:
                # interpret as row, col
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(arr.shape[0], arr.shape[1]),
                                 axes_pad=0.15,
                                 share_all=True,
                                 cbar_location='right',
                                 cbar_mode='single',
                                 cbar_size='7%',
                                 cbar_pad=0.15,
                                 )
                for ax, (i, j) in zip(grid, product(range(arr.shape[0]), range(arr.shape[1]))):
                    im = ax.imshow(arr[i, j], vmin=vmin, vmax=vmax)

                ax.cax.colorbar(im)
                ax.cax.toggle_label(True)
                tensorboard_logger.writer.add_figure("{}/{}".format(tensorboard_logger.key, key), fig, global_step=tensorboard_logger.global_step)

    tensorboard_logger.log_tensor = log_tensor

    def log_scalar(key: str, scalar_callback):
        calling_func = sys._getframe(1).f_code.co_name
        if tensorboard_logger._print_steps[calling_func] > 0 and (tensorboard_logger.global_step == 1 or tensorboard_logger.global_step % tensorboard_logger._print_steps[calling_func] == 0):
            tensorboard_logger.writer.add_scalar("{}/{}".format(tensorboard_logger.key, key), scalar_callback(), global_step=tensorboard_logger.global_step)

    tensorboard_logger.log_scalar = log_scalar

    def log_figure(key: str, figure_callback):
        calling_func = sys._getframe(1).f_code.co_name
        if tensorboard_logger._print_steps[calling_func] > 0 and (tensorboard_logger.global_step == 1 or tensorboard_logger.global_step % tensorboard_logger._print_steps[calling_func] == 0):
            tensorboard_logger.writer.add_figure("{}/{}".format(tensorboard_logger.key, key), figure_callback(), global_step=tensorboard_logger.global_step)

    tensorboard_logger.log_figure = log_figure

    def log_text(key: str, text_callback):
        calling_func = sys._getframe(1).f_code.co_name
        if tensorboard_logger._print_steps[calling_func] > 0 and (tensorboard_logger.global_step == 1 or tensorboard_logger.global_step % tensorboard_logger._print_steps[calling_func] == 0):
            tensorboard_logger.writer.add_text("{}/{}".format(tensorboard_logger.key, key), text_callback(), global_step=tensorboard_logger.global_step)

    tensorboard_logger.log_text = log_text

    def log_mesh(key: str, mesh_callback):
        calling_func = sys._getframe(1).f_code.co_name
        if tensorboard_logger._print_steps[calling_func] > 0 and (tensorboard_logger.global_step == 1 or tensorboard_logger.global_step % tensorboard_logger._print_steps[calling_func] == 0):
            vertices, colors, faces = mesh_callback()
            tensorboard_logger.writer.add_mesh("{}/{}".format(tensorboard_logger.key, key), vertices=vertices, colors=colors, faces=faces, global_step=tensorboard_logger.global_step)

    tensorboard_logger.log_mesh = log_mesh

    def real_tensorboard_logger(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # save the number of print_steps for this function
            tensorboard_logger._print_steps[func.__name__] = print_step

            if print_memory:
                tensorboard_logger.writer.add_scalar("{}/GPU_Memory_Allocated".format(tensorboard_logger.key), th.cuda.memory_allocated() * 2.**(-30), tensorboard_logger.global_step)
                tensorboard_logger.writer.add_scalar("{}/GPU_Memory_Allocated_Peak".format(tensorboard_logger.key), th.cuda.max_memory_allocated() * 2.**(-30), tensorboard_logger.global_step)
                tensorboard_logger.writer.add_scalar("{}/GPU_Memory_Reserved".format(tensorboard_logger.key), th.cuda.memory_reserved() * 2.**(-30), tensorboard_logger.global_step)
                tensorboard_logger.writer.add_scalar("{}/GPU_Memory_Reserved_Peak".format(tensorboard_logger.key), th.cuda.max_memory_reserved() * 2.**(-30), tensorboard_logger.global_step)

            # execute function
            return func(*args, **kwargs)
        return wrapper
    return real_tensorboard_logger
