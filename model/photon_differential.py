import torch as th
import typing as tp
from .utils import dot_product
from PhotonDifferentialSplatting import pds_forward, pds_backward


class PDS(th.autograd.Function):
    @staticmethod
    def forward(ctx, Ep: th.Tensor, xp: th.Tensor, Mp: th.Tensor, cp: th.Tensor, radius: th.Tensor, output_size: tp.Tuple, max_pixel_radius: int):
        ctx.save_for_backward(Ep, xp, Mp, cp, radius)
        ctx.max_pixel_radius = max_pixel_radius
        pds_grid = pds_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)[0]
        return pds_grid

    @staticmethod
    def backward(ctx, grad_pds: th.Tensor):
        Ep, xp, Mp, cp, radius = ctx.saved_tensors
        grad_Ep, grad_xp, grad_Mp = pds_backward(grad_pds, Ep, xp, Mp, cp, radius, ctx.max_pixel_radius)
        return grad_Ep, grad_xp, grad_Mp, None, None, None, None


class Photondifferential:
    flux = None
    position = None
    length = None
    direction = None
    normal = None
    Du = None
    Dv = None
    Dtheta = None
    Dphi = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def split(self, mask):
        mask.rename_(None)
        not_mask = th.logical_not(mask)

        not_dict = {k: v.rename(None)[not_mask].rename(*v.names) for k, v in self.__dict__.items() if v is not None}
        self.__dict__ = {k: v.rename(None)[mask].rename(*v.names) for k, v in self.__dict__.items() if v is not None}

        return Photondifferential(**not_dict)

    @classmethod
    def merge(self, pd_list: tp.List, dim='sample'):
        pd = Photondifferential()
        for k, v in pd_list[0].__dict__.items():
            pd.__dict__[k] = th.cat(tuple(p.__dict__[k] for p in pd_list), dim=dim)

        return pd

    def advance_path(self, compute_normals_at_hit: tp.Callable[[th.Tensor], th.Tensor]):
        hit_mask = self.length.ge(0)

        # restrict computation to stuff that actually hit something
        not_hit = self.split(hit_mask)

        self.position += self.length.align_as(self.position) * self.direction

        # compute new normal at hit
        self.normal, self.shading_normal = compute_normals_at_hit(self.object_index.rename(None), self.tri_index.rename(None), self.uv.rename(None))

        # with new normal and old direction update the photon differential
        self._advance_positional_differential()

        return hit_mask, not_hit

    def _advance_positional_differential(self, advance_differential_directions: tp.Callable[[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor], None] = None):
        # from Frisvad2014: Photon Differential Splatting for Rendering Caustics
        denominator = dot_product(self.normal, self.direction, dim='dim', keepdim=True)
        Du_hit = self.Du - (dot_product(self.normal, self.Du, dim='dim', keepdim=True) / denominator) * self.direction
        Dv_hit = self.Dv - (dot_product(self.normal, self.Dv, dim='dim', keepdim=True) / denominator) * self.direction
        self.Dtheta_hit = self.length.align_as(self.Dtheta) * (self.Dtheta - (dot_product(self.normal, self.Dtheta, dim='dim', keepdim=True) / denominator) * self.direction)
        self.Dphi_hit = self.length.align_as(self.Dphi) * (self.Dphi - (dot_product(self.normal, self.Dphi, dim='dim', keepdim=True) / denominator) * self.direction)

        self.Du = Du_hit + self.Dtheta_hit
        self.Dv = Dv_hit + self.Dphi_hit
