import torch as th
import torch.nn.functional as F
import PyOptix as popt
import math
import typing as tp
from collections import defaultdict

from .utils import dot_product, normalize_tensor, compute_orthonormal_basis, subtended_angle, slerp
from .photon_differential import Photondifferential
from .photon_differential import PDS as pds


def fused_silica(wavelength):
    '''
    Parameters:
    wavelength (Float): the wavelength in µm

    Fitted from data in [0.21µm, 6.7µm]

    Returns:
    Float: Refractive index
    '''
    return (1 + 0.6961663 / (1 - (0.0684043 / wavelength)**2) + 0.4079426 / (1 - (0.1162414 / wavelength)**2) + 0.8974794 / (1 - (9.896161 / wavelength)**2))**.5


def compute_point_light_dirs(height_offset: float, num_wavelengths: int, light_pos: th.Tensor, coords: th.Tensor, num_simul=1, num_inner_simul=1, smoothing=5, energy=1e-5):
    random_interpolation_coeffs = th.rand(coords.size('dim'), coords.size('height'), coords.size('width') * num_inner_simul, dtype=coords.dtype, device=coords.device, names=coords.names)

    bMin = coords.flatten(['height', 'width'], 'values').min(dim='values')[0].rename(None)
    bMax = coords.flatten(['height', 'width'], 'values').max(dim='values')[0].rename(None)
    upper_plane = th.full((2, 3, 3), height_offset, dtype=coords.dtype, device=coords.device)
    upper_plane[0, 0, :2] = bMin  # min corner first triangle
    upper_plane[0, 1, :2] = th.as_tensor([bMax[0], bMin[1]])
    upper_plane[0, 2, :2] = th.as_tensor([bMin[0], bMax[1]])

    upper_plane[1, 0, :2] = th.as_tensor([bMin[0], bMax[1]])
    upper_plane[1, 1, :2] = th.as_tensor([bMax[0], bMin[1]])
    upper_plane[1, 2, :2] = bMax
    upper_plane.rename_('triangle', 'vertex', 'dim')

    polar_corners = normalize_tensor((upper_plane - light_pos.align_as(upper_plane)).rename(None), dim=-1).rename('triangle', 'vertex', 'dim')

    # bilinear slerp
    lower = slerp(polar_corners[0, 0].align_to('height', 'width', 'dim'), polar_corners[0, 1].align_to('height', 'width', 'dim'), random_interpolation_coeffs[0].align_to('height', 'width', 'dim'))
    upper = slerp(polar_corners[1, 0].align_to('height', 'width', 'dim'), polar_corners[1, 2].align_to('height', 'width', 'dim'), random_interpolation_coeffs[0].align_to('height', 'width', 'dim'))
    directions = slerp(lower, upper, random_interpolation_coeffs[1].align_to('height', 'width', 'dim')).flatten(['height', 'width'], 'sample')

    # Du and Dv are zeros because of point light source
    Du = th.zeros_like(directions)
    Dv = th.zeros_like(directions)

    # fast orthonormal basis calculation, since we don't have a normal vector for point lights
    Dtheta, Dphi = compute_orthonormal_basis(directions.align_to('dim', 'sample'), dim='dim')
    Dtheta = Dtheta.align_to('sample', 'dim')
    Dphi = Dphi.align_to('sample', 'dim')
    sbt_angle = subtended_angle(light_pos.align_to('sample', 'dim'), upper_plane)

    total_photon_count = random_interpolation_coeffs.size('height') * random_interpolation_coeffs.size('width') * num_simul * num_wavelengths

    Dtheta *= 2 * smoothing * math.sqrt(sbt_angle.item() / (math.pi * total_photon_count))
    Dphi *= 2 * smoothing * math.sqrt(sbt_angle.item() / (math.pi * total_photon_count))

    # repeat tensors for each wavelength
    return Photondifferential(flux=th.full((num_wavelengths * directions.size('sample'),), energy / (num_simul * num_inner_simul), device=directions.device, dtype=directions.dtype).refine_names('sample'),
                              position=light_pos.align_to('sample', 'dim').rename(None).repeat(num_wavelengths * directions.size('sample'), 1).rename('sample', 'dim'),
                              direction=directions.rename(None).repeat_interleave(num_wavelengths, 0).rename('sample', 'dim'),
                              Du=Du.rename(None).repeat_interleave(num_wavelengths, 0).rename('sample', 'dim'),
                              Dv=Dv.rename(None).repeat_interleave(num_wavelengths, 0).rename('sample', 'dim'),
                              Dtheta=Dtheta.rename(None).repeat_interleave(num_wavelengths, 0).rename('sample', 'dim'),
                              Dphi=Dphi.rename(None).repeat_interleave(num_wavelengths, 0).rename('sample', 'dim'))


def generate_from_point_light(num_wavelengths: int, light_pos: th.Tensor, coords: th.Tensor, heights: th.Tensor, normals: th.Tensor, num_simul=32, smoothing=1):
    # ray differentials from Frisvad2014: Photon Differential Splatting for Rendering Caustics
    h, w = coords.size('height'), coords.size('width')
    # random sample each pixel on the surface
    # multiplied by pixel width, height
    random_offsets = (th.rand_like(coords) - 0.5) * (2 / th.tensor([w - 1, h - 1], names=('dim',), dtype=coords.dtype, device=coords.device)).align_as(coords)

    # sample with bilinear interpolation
    sample_pos = coords + random_offsets
    heights_at_sample = F.grid_sample(heights.rename(None)[(None, ) * 2], sample_pos.align_to('height', 'width', 'dim').rename(None).unsqueeze(0), align_corners=False, mode="bilinear", padding_mode="border")[0, 0].refine_names('height', 'width')
    normals_at_sample = normalize_tensor(F.grid_sample(normals.rename(None).unsqueeze(0), sample_pos.align_to('height', 'width', 'dim').rename(None).unsqueeze(0),
                                                       align_corners=False, mode="bilinear", padding_mode="border").squeeze(0), dim=0).refine_names('dim', 'height', 'width')

    pos_at_sample = th.cat((sample_pos, heights_at_sample.align_as(sample_pos)), dim='dim')
    dir_tensor = pos_at_sample - light_pos.align_as(pos_at_sample)
    length_tensor = th.norm(dir_tensor.rename(None), p=2, dim=0).refine_names('height', 'width')
    dir_tensor = normalize_tensor(dir_tensor.rename(None), p=2, dim=0).refine_names('dim', 'height', 'width')

    # Du and Dv are zeros, because of point light source
    Du = th.zeros_like(pos_at_sample)
    Dv = th.zeros_like(pos_at_sample)

    # we don't have a normal vector for point lights, so we compute a fast orthonormal basis for the direction vector of emitted light
    Dtheta, Dphi = compute_orthonormal_basis(dir_tensor, dim='dim')
    # make tensor of two triangles with extent of simulation space
    bMin = coords.flatten(['height', 'width'], 'values').min(dim='values')[0].rename(None)
    bMax = coords.flatten(['height', 'width'], 'values').max(dim='values')[0].rename(None)
    extents = th.zeros((2, 3, 3), dtype=coords.dtype, device=coords.device)
    extents[0, 0, :2] = bMin  # min corner first triangle
    extents[0, 1, :2] = th.as_tensor([bMax[0], bMin[1]])
    extents[0, 2, :2] = th.as_tensor([bMin[0], bMax[1]])

    extents[1, 2, :2] = bMax  # max corner second triangle
    extents[1, 0, :2] = th.as_tensor([bMin[0], bMax[1]])
    extents[1, 1, :2] = th.as_tensor([bMax[0], bMin[1]])
    sbt_angle = subtended_angle(light_pos.align_to('sample', 'dim'), extents.refine_names('triangle', 'vertex', 'dim'))

    total_photon_count = h * w * num_simul * num_wavelengths

    Dtheta *= 2 * smoothing * math.sqrt(sbt_angle.item() / (math.pi * total_photon_count))
    Dphi *= 2 * smoothing * math.sqrt(sbt_angle.item() / (math.pi * total_photon_count))

    # isotrope (=1) flux, equal over all wavelengths, distributed over num_simul photons
    pd = Photondifferential(flux=th.full_like(heights, 1e-5 / num_simul), position=pos_at_sample, length=length_tensor, direction=dir_tensor, Du=Du, Dv=Dv, Dtheta=Dtheta, Dphi=Dphi, normal=normals_at_sample)
    pd.advance_differential()

    # this assumes no occlusion occurs before the intersection at (sample_pos, heights_at_sample)
    return pd


def refract(incident_dirs: th.Tensor, normals: th.Tensor, iors: th.Tensor):
    # taken from: https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
    cosi = dot_product(incident_dirs, normals, dim='dim', normal=True, keepdim=True).rename(dim='channel')
    etai = th.ones_like(cosi) * th.ones_like(iors).align_as(cosi)
    etat = th.ones_like(cosi) * iors.align_as(etai)
    n = normals.clone()

    zero_mask = cosi.ge(0)
    # indexing not yet supported with named tensors :(
    cosi.rename_(None), zero_mask.rename_(None), etai.rename_(None), etat.rename_(None), n.rename_(None)
    cosi[th.logical_not(zero_mask)] = -cosi[th.logical_not(zero_mask)]
    etai[zero_mask.expand_as(etai)], etat[zero_mask.expand_as(etat)] = etat[zero_mask.expand_as(etat)], etai[zero_mask.expand_as(etai)]
    n[zero_mask.expand_as(n)] = -n[zero_mask.expand_as(n)]

    eta = etai / etat
    k = 1 - eta**2 * (1 - cosi**2)

    n_expand = n.unsqueeze(1).expand(-1, iors.size('channel'), *((-1,) * (n.dim() - 1)))
    incident_dirs_expand = incident_dirs.rename(None).unsqueeze(1).expand(-1, iors.size('channel'), *((-1, ) * (incident_dirs.dim() - 1)))
    cosi_expand = cosi.expand_as(k)

    # mask negative sqrt -> total reflection
    rm = k.ge(0)
    # refracted_vec = th.zeros_like(n_expand)
    # refracted_vec[:, rm] = eta[rm] * incident_dirs_expand[:, rm] + (eta[rm] * cosi_expand[rm] - th.sqrt(k[rm])) * n_expand[:, rm]
    refracted_vec = eta[rm] * incident_dirs_expand[:, rm] + (eta[rm] * cosi_expand[rm] - th.sqrt(k[rm])) * n_expand[:, rm]
    return refracted_vec.refine_names('dim', 'sample'), rm.refine_names('channel', 'height', 'width')


def refract_reflect_differentials(incident_dirs: th.Tensor,
                                  normals: th.Tensor,
                                  Dtheta: th.Tensor,
                                  Dphi: th.Tensor,
                                  dN_dtheta: th.Tensor,
                                  dN_dphi: th.Tensor,
                                  iors_outer: th.Tensor,
                                  iors_inner: th.Tensor):
    cosi = dot_product(incident_dirs, normals, dim=-1, normal=True, keepdim=True)  # -cosi for Glassner

    etai = iors_outer.clone()
    etat = iors_inner.clone()
    N = normals.clone()
    dNdt = dN_dtheta.clone()
    dNdp = dN_dphi.clone()

    zero_mask = cosi.gt(0).squeeze(-1)
    if zero_mask.any():
        # flip refractive indices and normal as well as normal derivatives
        etai[zero_mask], etat[zero_mask] = etat[zero_mask], etai[zero_mask]
        N[zero_mask] = -normals[zero_mask]
        dNdt[zero_mask] = -dN_dtheta[zero_mask]
        dNdp[zero_mask] = -dN_dphi[zero_mask]

    eta = (etai / etat).unsqueeze(-1)
    k = 1 - eta**2 * (1 - cosi**2)

    # total internal reflection cases
    rm = k.gt(0).squeeze(-1)
    if rm.all():
        neg_sq = -th.sqrt(k)
        coso = eta * cosi - neg_sq

        # real refraction case
        outgoing_dir = eta * incident_dirs - coso * N
        # Igehy1999: p4
        Dtheta_out = eta * Dtheta - (coso * dNdt + ((eta - (eta**2 * cosi) / neg_sq) * (dot_product(Dtheta, N, keepdim=True) + dot_product(incident_dirs, dNdt, keepdim=True))) * N)
        Dphi_out = eta * Dphi - (coso * dNdp + ((eta - (eta**2 * cosi) / neg_sq) * (dot_product(Dphi, N, keepdim=True) + dot_product(incident_dirs, dNdp, keepdim=True))) * N)

    else:
        # do the same thing as above, but with masking
        outgoing_dir = th.zeros_like(incident_dirs)
        Dtheta_out = th.zeros_like(Dtheta)
        Dphi_out = th.zeros_like(Dphi)

        neg_sq = -th.sqrt(k[rm])
        coso = eta[rm] * cosi[rm] - neg_sq

        outgoing_dir[rm] = eta[rm] * incident_dirs[rm] - coso * N[rm]
        # Igehy1999: p4
        Dtheta_out[rm] = eta[rm] * Dtheta[rm] - (coso * dNdt[rm] + ((eta[rm] - (eta[rm]**2 * cosi[rm]) / neg_sq) * (dot_product(Dtheta[rm], N[rm], keepdim=True) + dot_product(incident_dirs[rm], dNdt[rm], keepdim=True))) * N[rm])
        Dphi_out[rm] = eta[rm] * Dphi[rm] - (coso * dNdp[rm] + ((eta[rm] - (eta[rm]**2 * cosi[rm]) / neg_sq) * (dot_product(Dphi[rm], N[rm], keepdim=True) + dot_product(incident_dirs[rm], dNdp[rm], keepdim=True))) * N[rm])

        n_rm = th.logical_not(rm)
        # mask negative sqrt -> total internal reflection (r = d - 2(d.n)n)
        outgoing_dir[n_rm] = incident_dirs[n_rm] - 2 * cosi[n_rm] * N[n_rm]
        # Igehy1999: p3
        Dtheta_out[n_rm] = Dtheta[n_rm] - 2 * (cosi[n_rm] * dNdt[n_rm] + (dot_product(Dtheta[n_rm], N[n_rm], keepdim=True) + dot_product(incident_dirs[n_rm], dNdt[n_rm], keepdim=True)) * N[n_rm])
        Dphi_out[n_rm] = Dphi[n_rm] - 2 * (cosi[n_rm] * dNdp[n_rm] + (dot_product(Dphi[n_rm], N[n_rm], keepdim=True) + dot_product(incident_dirs[n_rm], dNdp[n_rm], keepdim=True)) * N[n_rm])

    return outgoing_dir, Dtheta_out, Dphi_out


def get_normal_from_height_field(height_tensor: th.Tensor, element_size: th.Tensor, normalize=True):
    # this is a simple sobel filter https://de.wikipedia.org/wiki/Sobel-Operator
    sobel = th.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], names=('dim', 'input_dim', 'kernel_height', 'kernel_width'), dtype=height_tensor.dtype, device=height_tensor.device)
    sobel *= -1 / (8 * element_size).align_to('dim', 'input_dim', 'kernel_height', 'kernel_width')
    conv = th.cat((F.conv2d(F.pad(height_tensor.rename(None)[(None, ) * 2], (1, 1, 1, 1), mode='replicate'), sobel.rename(None)).squeeze(0), th.ones(1, *height_tensor.size(), dtype=height_tensor.dtype, device=height_tensor.device)), dim=0)

    return normalize_tensor(conv, dim=0).refine_names('dim', *height_tensor.names) if normalize else conv.refine_names('dim', *height_tensor.names)


def change_of_basis_matrix(pd: Photondifferential):
    tmp = th.cross(pd.Dv, pd.shading_normal, dim=0)
    denominator = dot_product(pd.Du, tmp, dim=0)

    non_degenerate_mask = th.logical_not(denominator.eq(0))
    factor = 2 / denominator[non_degenerate_mask]
    return factor * th.stack((tmp[:, non_degenerate_mask], th.cross(pd.shading_normal[:, non_degenerate_mask], pd.Du[:, non_degenerate_mask], dim=0)), dim=0).refine_names('row', 'dim', 'sample'), non_degenerate_mask


def splat_photons(pd: Photondifferential, normalized_coords: th.Tensor, output_size, max_pixel_radius=20):
    Mp, good_mask = change_of_basis_matrix(pd)

    radius = 0.5 * th.max(th.norm(pd.Du[:, good_mask], dim=0), th.norm(pd.Dv[:, good_mask], dim=0))
    Ep = pd.flux[good_mask] / th.norm(th.cross(pd.Du[:, good_mask], pd.Dv[:, good_mask], dim=0), dim=0) * 4  # * math.pi # removed because of multiplication in pds sum

    return pds.apply(Ep, normalized_coords[:, good_mask], Mp, pd.channel_coords[good_mask].unsqueeze(0), radius, output_size, max_pixel_radius)


def compute_recursive_refraction(iors: th.Tensor,
                                 photon_map_size: tp.Tuple,
                                 max_pixel_radius: int,
                                 compute_normals_at_hit: tp.Callable[[th.Tensor, th.Tensor, th.Tensor], defaultdict],
                                 compute_differential_normals_at_hit: tp.Callable[[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor], tp.Tuple[th.Tensor, th.Tensor]],
                                 pd: Photondifferential,
                                 ground_plane_index=1):

    pd.iors_outer = th.ones_like(iors).rename(None).repeat(pd.position.size('sample') // iors.size('channel')).rename('sample')
    pd.iors_inner = iors.rename(None).repeat(pd.position.size('sample') // iors.size('channel')).rename('sample')
    pd.channel_coords = th.arange(0, iors.size('channel'), dtype=th.int64, device=iors.device).repeat(pd.position.size('sample') // iors.size('channel')).rename('sample')

    ground_plane_pds = []
    while pd.position.size('sample') > 0:
        pd.length, pd.uv, pd.object_index, pd.tri_index = popt.trace_rays(pd.position, pd.direction, 0)

        # filter out stuff that hasn't hit anything
        pd.length.rename_('sample')
        pd.object_index.rename_('sample')
        pd.tri_index.rename_('sample')
        pd.uv.rename_('sample', 'dim')

        # advance path and positional differentials
        hit_mask, _ = pd.advance_path(compute_normals_at_hit)

        # filter out ground plane hits
        ground_plane_pds.append(pd.split(th.logical_not(pd.object_index.eq(ground_plane_index))))

        # compute the differential normals
        dN_dtheta, dN_dphi = compute_differential_normals_at_hit(pd.object_index.rename(None), pd.tri_index.rename(None), pd.shading_normal.rename(None), pd.Dtheta_hit.rename(None), pd.Dphi_hit.rename(None))
        dN_dtheta.rename_('sample', 'dim')
        dN_dphi.rename_('sample', 'dim')

        # compute refraction direction, and new differential directions
        pd.direction, pd.Dtheta, pd.Dphi = refract_reflect_differentials(pd.direction.rename(None), pd.shading_normal.rename(None), pd.Dtheta.rename(None), pd.Dphi.rename(None),
                                                                         dN_dtheta.rename(None), dN_dphi.rename(None), pd.iors_outer.rename(None), pd.iors_inner.rename(None))
        pd.direction.rename_('sample', 'dim')
        pd.Dtheta.rename_('sample', 'dim')
        pd.Dphi.rename_('sample', 'dim')

    pd = Photondifferential.merge(ground_plane_pds)

    # TODO: pd.position should be pd.tex_coord for more general application
    # reshuffle order for splat_photons
    pd.Du = pd.Du.align_to('dim', 'sample').rename(None)
    pd.Dv = pd.Dv.align_to('dim', 'sample').rename(None)
    pd.shading_normal = pd.shading_normal.align_to('dim', 'sample').contiguous().rename(None)
    pd.channel_coords.rename_(None)
    pd.flux.rename_(None)
    return splat_photons(pd, pd.position.align_to('dim', 'sample').rename(None)[:2], (iors.size('channel'), *photon_map_size), max_pixel_radius=max_pixel_radius).refine_names('channel', 'height', 'width')


def compute_refraction(coords, height_field, photon_map_size, iors, compute_incident_dirs):
    # photon differentials from Frisvad2014: Photon Differential Splatting for Rendering Caustics
    element_size = (coords[:, 1, 1] - coords[:, 0, 0])

    # compute the normal map
    normals = get_normal_from_height_field(height_field, element_size)

    # randomly generate sample directions
    pd = compute_incident_dirs(coords, height_field, normals)

    # here we assume that every ray from infinity to cell (i,j) also intersects the surface at (i,j,meanWaterHeight)
    # this is of course pretty wrong as height changes locally and occlusion from far away cells can block rays shot at (i,j) from ever hitting this cell
    # ideally one would do a general 'intersect(ray_origin, ray_direction, scene)' function, but because of occlusions this is not easily differentiable in the general case
    outgoing_dir, refraction_mask = refract(pd.direction, pd.normal, iors)
    rm_names = refraction_mask.names
    refraction_mask.rename_(None)

    # Compute the point, where the refracted ray deposits its' energy. This is simple, because the ground plane is flat (and assumed to be at 0 height). In the general case this would be another call
    # to the intersect function.
    pos_at_sample = pd.position.align_to('dim', 'channel', 'height', 'width').expand(-1, iors.size('channel'), -1, -1).rename(None)[:, refraction_mask].refine_names('dim', 'sample')
    bottom_intersection = th.zeros_like(pos_at_sample).rename(None)
    intersection_length = pos_at_sample.select('dim', 2) / outgoing_dir.select('dim', 2)
    bottom_intersection = pos_at_sample - intersection_length * outgoing_dir

    # get the indices from bottom_intersection (assuming again coordinate range (-1; 1) for valid positions)
    normalized_coords = bottom_intersection.rename(None)[:2]
    # create an mask for indices in the correct range
    in_bounds_mask = (normalized_coords > -1).all(dim=0) & (normalized_coords < 1).all(dim=0)
    normalized_coords = normalized_coords[:, in_bounds_mask].refine_names('dim', 'sample')
    # calculate channel coordinate as third dim
    channel_coords = th.arange(0, iors.size('channel'), dtype=th.int64, device=pd.flux.device)[:, None, None].expand_as(refraction_mask)[refraction_mask][in_bounds_mask].refine_names('sample')

    # update photon differentials
    pd.position = bottom_intersection.rename(None)[:, in_bounds_mask].refine_names('dim', 'sample')
    pd.flux = pd.flux.align_to('channel', 'height', 'width').expand(iors.size('channel'), -1, -1).rename(None)[refraction_mask][in_bounds_mask].refine_names('sample')
    pd.length = intersection_length.rename(None)[in_bounds_mask].refine_names('sample')
    pd.direction = outgoing_dir.rename(None)[:, in_bounds_mask].refine_names('dim', 'sample')
    pd.Du = pd.Du.align_to('dim', 'channel', 'height', 'width').expand(-1, iors.size('channel'), -1, -1).rename(None)[:, refraction_mask][:, in_bounds_mask].refine_names('dim', 'sample')
    pd.Dv = pd.Dv.align_to('dim', 'channel', 'height', 'width').expand(-1, iors.size('channel'), -1, -1).rename(None)[:, refraction_mask][:, in_bounds_mask].refine_names('dim', 'sample')
    pd.Dtheta = pd.Dtheta.align_to('dim', 'channel', 'height', 'width').expand(-1, iors.size('channel'), -1, -1).rename(None)[:, refraction_mask][:, in_bounds_mask].refine_names('dim', 'sample')
    pd.Dphi = pd.Dphi.align_to('dim', 'channel', 'height', 'width').expand(-1, iors.size('channel'), -1, -1).rename(None)[:, refraction_mask][:, in_bounds_mask].refine_names('dim', 'sample')
    pd.normal = th.tensor([[0], [0], [1]], dtype=pos_at_sample.dtype, device=pos_at_sample.device, names=('dim', 'sample'))  # normal vector at hit point is simply positive z-Direction

    refraction_mask.rename_(*rm_names)
    in_bounds_mask.rename_('sample')

    pd.advance_differential()

    # splat (i.e. distribute to surrounding cells) the energy into a texture at bottom_intersection
    return splat_photons(pd, normalized_coords, channel_coords, (iors.size('channel'), *photon_map_size)).refine_names('channel', 'height', 'width')
