import torch as th
from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
from model.utils import tensorboard_logger as tbl, height_field_to_mesh
from model.caustics import compute_recursive_refraction, compute_point_light_dirs, fused_silica, get_normal_from_height_field
from model.renderable_object import create_from_height_field
from hyperparameter_helper import get_argument_set
import os
import sys

__authors__ = "Marc Kassubeck, Florian Buergel"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Marc Kassubeck"
__email__ = "kassubeck@cg.cs.tu-bs.de"
__status__ = "Development"


def post_optimization_plot(args, coords: np.ndarray, height_field: np.ndarray, height_field_recon: np.ndarray, simulation: np.ndarray):
    hclim1 = np.min([height_field.min(), height_field_recon.min()])
    hclim2 = np.max([height_field.max(), height_field_recon.max()])

    sclim1 = simulation.min()
    sclim2 = simulation.max()

    plt.figure(0)
    r = 2
    c = 4

    plt.subplot2grid((r, c), (0, 0)), plt.imshow(height_field, vmin=hclim1, vmax=hclim2), plt.title("Height Field (Exact)"), plt.colorbar(), plt.clim(hclim1, hclim2)
    if args.read_gt is None:
        plt.subplot2grid((r, c), (0, 1)), plt.imshow(height_field_recon, vmin=hclim1, vmax=hclim2), plt.title("Height Field (Recon.)"), plt.colorbar(), plt.clim(hclim1, hclim2)
    else:
        plt.subplot2grid((r, c), (0, 1)), plt.imshow(height_field_recon), plt.title("Height Field (Recon.)"), plt.colorbar()  # no specific limits

    plt.subplot2grid((r, c), (1, 0))
    plt.plot(coords[0, args.height_field_resolution // 2], height_field[args.height_field_resolution // 2], '-b')
    plt.plot(coords[0, args.height_field_resolution // 2], height_field_recon[args.height_field_resolution // 2], '--r')
    plt.title("Center Slice (exact (-b), reconstructed (--r))")

    if simulation.shape[0] >= 3:
        plt.subplot2grid((r, c), (1, 1)), plt.imshow(simulation[0], vmin=sclim1, vmax=sclim2), plt.title("Simulation ch.1"), plt.colorbar(), plt.clim(sclim1, sclim2)
        plt.subplot2grid((r, c), (1, 2)), plt.imshow(simulation[1], vmin=sclim1, vmax=sclim2), plt.title("Simulation ch.2"), plt.colorbar(), plt.clim(sclim1, sclim2)
        plt.subplot2grid((r, c), (1, 3)), plt.imshow(simulation[2], vmin=sclim1, vmax=sclim2), plt.title("Simulation ch.3"), plt.colorbar(), plt.clim(sclim1, sclim2)
    else:
        plt.subplot2grid((r, c), (1, 1)), plt.imshow(simulation if simulation.ndim == 2 else simulation[0], vmin=sclim1, vmax=sclim2), plt.title("Simulation"), plt.colorbar(), plt.clim(sclim1, sclim2)

    plt.savefig(os.path.join('runs', '{}.png'.format(args.datetime)), format='png', dpi=600)


def gauss(sigma, mu, x):
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * th.exp(-0.5 * ((x - mu) / sigma)**2)
    return pdf


def twovariate_gauss(sig1, sig2, mu1, mu2, rho, x1, x2):
    pdf = 1 / (2 * np.pi * sig1 * sig2 * np.sqrt(1 - rho * rho)) * th.exp(-1 / (2 * (1 - rho * rho)) * ((x1 - mu1)**2 / (sig1**2) + (x2 - mu2)**2 / (sig2**2) - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sig1 * sig2)))
    return pdf


def line_h(sigma, borderL, borderR, x1, x2):
    pdf1 = gauss(sigma, 0, x2)
    pdf1left = twovariate_gauss(sigma, sigma, borderL, 0, 0, x1, x2)
    pdf1right = twovariate_gauss(sigma, sigma, borderR, 0, 0, x1, x2)

    mask1 = (borderL < x1)
    mask2 = (x1 < borderR)
    mask3 = (x1 <= borderL)
    mask4 = (borderR <= x1)

    scale = th.max(pdf1) / th.max(pdf1left)
    pdf = pdf1 * mask1 * mask2 + scale * pdf1left * mask3 + scale * pdf1right * mask4
    return pdf


def line_v(sigma, borderB, borderT, x1, x2):
    pdf1 = gauss(sigma, 0, x1)
    pdf1top = twovariate_gauss(sigma, sigma, 0, borderT, 0, x1, x2)
    pdf1bottom = twovariate_gauss(sigma, sigma, 0, borderB, 0, x1, x2)

    mask1 = (borderT > x2)
    mask2 = (x2 > borderB)
    mask3 = (x2 >= borderT)
    mask4 = (borderB >= x2)

    scale = th.max(pdf1) / th.max(pdf1top)
    pdf = pdf1 * mask1 * mask2 + scale * pdf1top * mask3 + scale * pdf1bottom * mask4
    return pdf


def generate_height_field(args, coords: th.Tensor):
    if args.height_field_option == 'gaussian':
        x = 6 * coords.select('dim', 0)
        pdf = gauss(1, 0, x)
        return 0.5 * pdf + args.height_offset
    elif args.height_field_option == 'image':
        img = th.from_numpy(np.array(Image.open("inputHeightField/eg_logo.png").resize((args.height_field_resolution, args.height_field_resolution)))).to(args.dtype) / 255.
        return (img * (args.upper_bound - args.lower_bound) + args.height_offset).to(args.device).refine_names('height', 'width')  # convert numpy array to torch tensor in GPU
    elif args.height_field_option == 'gaussian_damage':
        sigma = 1
        mu = 0
        x = 6 * coords.select('dim', 0)
        pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * th.exp(-0.5 * ((x - mu) / sigma)**2)
        n0 = pdf.size()[0]
        n1 = pdf.size()[1]
        nPixel = 0.05 * n0  # procent of n0
        idxDam0a = int(np.ceil(n0 / 3))  # damage 0 start
        idxDam0b = int(np.ceil(n0 / 3 + nPixel))  # damage 0 end
        nPixel = 0.01 * n0
        idxDam1a = int(np.ceil(n0 / 3 * 2))  # damage 1 start
        idxDam1b = int(np.ceil(n0 / 3 * 2 + nPixel))  # damage 1 end
        pdf[idxDam0a:idxDam0b, :] = 0
        pdf[idxDam1a:idxDam1b, :] = 0
        return 0.5 * pdf + args.height_offset
    elif args.height_field_option == 'gaussian_two':
        x1 = 6 * coords.select('dim', 0)
        x2 = 6 * coords.select('dim', 1)
        pdf1 = 2 * twovariate_gauss(2, 0.75, 0, 4, 0, x1, x2)  # bottom, sig1 = 1 would be similar to height field "gaussian"
        pdf2 = twovariate_gauss(0.5, 2, -4, 0, 0, x1, x2)  # left
        pdf3 = twovariate_gauss(1, 0.5, 0, 0, 0, x1, x2)  # center
        pdf4 = twovariate_gauss(1, 0.5, 3, 0, 0, x1, x2)  # right
        return 0.5 * (pdf1 + pdf2 + pdf3 + pdf4) + args.height_offset
    elif args.height_field_option == 'print_lines':
        x1 = 6 * coords.select('dim', 0)
        x2 = 6 * coords.select('dim', 1)

        l1 = line_v(0.5, -4.5, +4.5, x1, x2)  # vertical line
        l2 = 0.75 * line_v(0.5, -4.5, +1.0, x1 - 3, x2)  # shift 3 to right
        l3 = 0.75 * line_v(0.5, 2.5, +4.5, x1 - 3, x2)  # shift 3 to right

        l4 = line_h(0.5, -4.5, -2.0, x1, x2)  # horizontal
        l5 = 0.5 * line_h(0.25, -4.5, -2.0, x1, x2 - 2.5)  # horizontal bottom 1
        l5scale = th.max(l1) / th.max(l5)  # scale l5 to l1
        l5 = l5scale * l5

        l6 = 0.25 * line_h(0.125, -4.5, -2.0, x1, x2 - 4.5)  # horizontal bottom 2 # sigma 0.125 too small for reconstruction
        l6scale = th.max(l1) / th.max(l6)  # scale l6 to l1
        ll = l6scale * l6

        return 0.1 * (l1 + l2 + l3 + l4 + l5 + l6) + args.height_offset
    elif args.height_field_option == 'oblique_lines':
        x1 = 6 * coords.select('dim', 0).rename(None)
        x2 = 6 * coords.select('dim', 1).rename(None)

        theta = np.radians(15)  # e.g. 15 deg (mathematical positive rotation direction)
        c = np.cos(theta)
        s = np.sin(theta)
        R = th.tensor([[c, -s], [s, c]]).to(args.device)

        x1flat = x1.reshape(-1)  # flatten x1
        x2flat = x2.reshape(-1)  # flatten x2
        x1x2 = th.stack((x1flat, x2flat))
        x1x2 = th.mm(R, x1x2)
        x1rot = th.reshape(x1x2[0], x1.shape)  # reshape
        x2rot = th.reshape(x1x2[1], x2.shape)  # reshape
        # rename
        x1rot.rename_('height', 'width')
        x2rot.rename_('height', 'width')

        x1 = x1rot
        x2 = x2rot

        # print lines on rotated grid (same as in print_lines)
        l1 = line_v(0.5, -4.5, +4.5, x1, x2)  # vertical line
        l2 = 0.75 * line_v(0.5, -4.5, +1.0, x1 - 3, x2)  # shift 3 to right
        l3 = 0.75 * line_v(0.5, 2.5, +4.5, x1 - 3, x2)  # shift 3 to right

        l4 = line_h(0.5, -4.5, -2.0, x1, x2)  # horizontal
        l5 = 0.5 * line_h(0.25, -4.5, -2.0, x1, x2 - 2.5)  # horizontal bottom 1
        l5scale = th.max(l1) / th.max(l5)  # scale l5 to l1
        l5 = l5scale * l5

        l6 = 0.25 * line_h(0.125, -4.5, -2.0, x1, x2 - 4.5)  # horizontal bottom 2 # sigma 0.125 too small for reconstruction
        l6scale = th.max(l1) / th.max(l6)  # scale l6 to l1
        ll = l6scale * l6

        return 0.1 * (l1 + l2 + l3 + l4 + l5 + l6) + args.height_offset


def extShrink(x, alphaShrink, a, b):
    # extended soft-shrinkage operator:
    zeros = th.zeros_like(x)
    ones = th.ones_like(x)
    out = th.max(th.min(th.max((th.abs(x) - alphaShrink), zeros) * th.sign(x), b * ones), a * ones)
    return out


def VconsFast(x, d1, d2):
    # proximal mapping for conservation of volume: \delta_{[d1,d2]} (\|d\|_1)
    # d1 and d2 are the constraints (volume/area)
    inf = float("inf")
    xnorm = th.norm(x, 1)
    # factor = 1/(n1*n2)  # factor to influence the setting in the case of constraints... (maybe 0 or 1/(n1*n2) or random between some bounds?)
    factor = 1
    #
    # Case 1: \|x\|_1 \leq d1:
    if xnorm <= d1:
        ind = th.where((d1 <= x) & (x < inf))
        x[ind[0][:], ind[1][:]] = -d1 * factor
        ind = th.where((-d1 < x) & (x < d1))
        x[ind[0][:], ind[1][:]] = 0
        ind = th.where((-inf < x) & (x <= -d1))
        x[ind[0][:], ind[1][:]] = d1 * factor

    # Case 2: \|x\|_1 \in (d1,d2):
    elif d1 < xnorm < d2:  # next: consider single values of the matrix to modify them
        # Nothing is done in the case of -d2 < x < d2
        pass

    # Case 3: \|x\|_1 \geq d2:
    elif xnorm >= d2:
        ind = th.where((-inf < x) & (x < 0))
        x[ind[0][:], ind[1][:]] = -d2 * factor
        ind = th.where(x == 0)
        x[ind[0][:], ind[1][:]] = 0
        ind = th.where((0 < x) & (x < inf))
        x[ind[0][:], ind[1][:]] = d2 * factor
    else:
        print("VconsFast: no case")

    return x


def VconsHeur(x, d1, d2, mask, gamma, volume_radius):
    # gamma = 0.05 # factor: conservation of volume heuristic
    n1 = x.size()[0]
    n2 = x.size()[1]
    vx = th.sum(x)  # sum instead of th.norm(x, 1) to take into account negative entries correctly
    xmean = F.avg_pool2d(x[None, None, :, :], 2 * volume_radius + 1, stride=1, padding=volume_radius)[0, 0]

    # idea of sin: slow growing at the beginng and end; fast in the middle
    if vx <= d1:
        x = x + mask * xmean * th.sin(vx / d1 * np.pi) * gamma
    elif vx >= d2:
        x = x - mask * xmean * th.sin((vx / d2 - 1) * np.pi) * gamma

    return x


def gradNeumann(x, area):
    # standard finite differences with Neumann boundary conditions (from Chambolle2011, Sec. 6.1)
    h = th.sqrt(area)
    gx1 = th.zeros_like(x)
    gx2 = th.zeros_like(x)

    gx1 = (th.roll(x, -1, 0) - x) / h
    gx1[-1, :] = 0

    gx2 = (th.roll(x, -1, 1) - x) / h
    gx2[:, -1] = 0

    return gx1, gx2


def div(x1, x2, area):
    # discrete divergence as in Buergel2017, Sec. 4.5 with correction in Buergel2019b, Sec. 7.3
    divx1 = th.zeros_like(x1)
    divx2 = th.zeros_like(x2)
    h = th.sqrt(area)

    # Compute divx1 = (div x)_{i,j}^{(1)}:
    divx1[0, :] = x1[0, :] / h
    divx1 = (x1 - th.roll(x1, -1, 0)) / h
    divx1[-1, :] = -x1[-1, :] / h

    divx2[0, :] = x2[0, :] / h
    divx2 = (x2 - th.roll(x2, -1, 1)) / h
    divx2[:, -1] = -x2[:, -1] / h

    divx = divx1 + divx2
    return divx


def normTV(gx1, gx2):
    # Precisely, TV is not a norm but a semi-norm.
    # normgx = \|\nabla x\|_2 = \sqrt{\|gx1\|_F^2 + \|gx2\|_F^2} (from Chambolle2011, Sec. 6.2)
    normgx = th.sqrt(th.norm(gx1, 'fro')**2 + th.norm(gx2, 'fro')**2)
    return normgx


def derivativeTV(x, area, epstv):  # derivative of total variation
    # derivative: - \div(\grad x / \|\grad x\|_2)
    gx1, gx2 = gradNeumann(x, area)  # \nabla x (gx1 and gx2 are zero if x is zero...)
    # normgx = normTV(gx1,gx2) # semi-norm TV (is 0 if x is zero...; therefore stabizlization version...)
    normgx = normTV(gx1 + epstv, gx2 + epstv)  # semi-norm TV with stabilization
    divx = div(gx1 / normgx, gx2 / normgx, area)  # discrete divergence: div (is nan if x is zero...)
    return -divx  # derivative


# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# decorated optimization loops

@tbl(print_step=10)
def optimization_loop_baseline(i, args, optim, coords, iors, light_pos, height_field_exact, height_field_recon_d, sns, sns_norm, update_scene, compute_normals_at_hit, compute_differential_normals_at_hit, mean_energy=1e-5):
    optim.zero_grad()

    height_field_recon = height_field_recon_d + args.height_offset
    update_scene(height_field_recon)

    simulation_recon = sum(compute_recursive_refraction(iors, (args.photon_map_size, args.photon_map_size), args.max_pixel_radius, compute_normals_at_hit, compute_differential_normals_at_hit,
                                                        compute_point_light_dirs(args.height_offset, iors.numel(), light_pos, coords, num_simul=args.num_simulations, num_inner_simul=args.num_inner_simulations, smoothing=args.splat_smoothing, energy=mean_energy))
                           for i in range(args.num_simulations))

    tbl.log_tensor('Simulation_Estimated', lambda: simulation_recon)
    err = (th.norm(height_field_exact.rename(None) - (height_field_recon_d.detach().rename(None) + args.height_offset), 2) / th.norm(height_field_exact.rename(None), 2)).item()  # relative error
    tbl.log_scalar('Error', lambda: err)

    fdis = args.objective_func(simulation_recon, sns)

    discrepancy = (th.sqrt(2 * fdis) / sns_norm).item()
    tbl.log_scalar('Discrepancy', lambda: discrepancy)

    # iterations: first i is 0; Iter 0 shows values before first reconstruction iteration
    if i <= 10 or i % 10 == 0 or i > args.num_iterations - 10:
        tqdm.write("Iter {}, dis {:2.4f}, err {:2.4f}".format(str(i).zfill(len(str(args.num_iterations))), discrepancy, err))

    fdis.backward()
    tbl.log_tensor('Height_Field_Gradient', lambda: height_field_recon_d.grad.data)

    # make a normal sgd step
    optim.step()

    def slice_figure():
        fig = plt.figure()
        plt.gca().set(aspect=1)
        plt.plot(coords.select('dim', 0).select('height', args.height_field_resolution // 2).cpu().numpy(), height_field_exact.select('height', args.height_field_resolution // 2).cpu().numpy(), '-b')
        plt.plot(coords.select('dim', 0).select('height', args.height_field_resolution // 2).cpu().numpy(), (height_field_recon_d + args.height_offset).detach().select('height', args.height_field_resolution // 2).cpu().numpy(), '--r')
        return fig
    tbl.log_figure("Center_Slice", slice_figure)

    if discrepancy <= args.tau_dis * args.noise_level:
        tqdm.write(" ")
        tqdm.write("Iter {}, dis {:2.4f}, err {:2.4f}".format(str(i).zfill(len(str(args.num_iterations))), discrepancy, err))  # output of last iteration
        return False, simulation_recon

    return True, simulation_recon


@tbl(print_step=10)
def optimization_loop_landweber_pixel(i, args, optim, coords, iors, light_pos, pixel_area, pixel_area_brightness, printing_volume, mask, height_field_exact, height_field_recon_d,
                                      sns, sns_norm, update_scene, compute_normals_at_hit, compute_differential_normals_at_hit, mean_energy=1e-5):
    optim.zero_grad()

    height_field_recon = height_field_recon_d + args.height_offset
    update_scene(height_field_recon)

    simulation_recon = sum(compute_recursive_refraction(iors, (args.photon_map_size, args.photon_map_size), args.max_pixel_radius, compute_normals_at_hit, compute_differential_normals_at_hit,
                                                        compute_point_light_dirs(args.height_offset, iors.numel(), light_pos, coords, num_simul=args.num_simulations, num_inner_simul=args.num_inner_simulations, smoothing=args.splat_smoothing, energy=mean_energy))
                           for i in range(args.num_simulations))

    tbl.log_tensor('Simulation_Estimated', lambda: simulation_recon)
    err = (th.norm(height_field_exact.rename(None) - (height_field_recon_d.detach().rename(None) + args.height_offset), 2) / th.norm(height_field_exact.rename(None), 2)).item()  # relative error
    tbl.log_scalar('Error', lambda: err)

    dorig = height_field_recon_d.data.rename(None)

    # compute discrepancy and penalty terms (of iteration before)
    fdis = args.objective_func(simulation_recon, sns)
    discrepancy = (th.sqrt(2 * fdis) / sns_norm).item()
    tbl.log_scalar('Discrepancy', lambda: discrepancy)

    fspa = args.alpha_pixel * th.norm(dorig, 1)
    gx1, gx2 = gradNeumann(dorig, pixel_area)
    ftv = args.beta_pixel * normTV(gx1, gx2)

    tbl.log_scalar('F_Sparsity_Pixel', lambda: fspa.item())
    tbl.log_scalar('F_Total_Variation', lambda: ftv.item())

    # iterations: first i is 0; Iter 0 shows values before first reconstruction iteration
    if i <= 10 or i % 10 == 0 or i > args.num_iterations - 10:
        tqdm.write("Iter {}, dis {:2.4f}, err {:2.4f}, fdis {:2.4f}, fspa {:2.4f}, ftv {:2.4f}".format(str(i).zfill(len(str(args.num_iterations))), discrepancy, err, fdis, fspa, ftv.item()))

    fdis.backward()
    derivativefdis = height_field_recon_d.grad.data

    tbl.log_tensor('Height_Field_Gradient', lambda: derivativefdis)

    # Veps = 0.05  # max. relative error of printed volume
    d1 = (printing_volume - printing_volume * args.volume_eps) / pixel_area
    d2 = (printing_volume + printing_volume * args.volume_eps) / pixel_area

    # a) gradient descent
    d = dorig - args.tau_pixel * derivativefdis  # d is changed afterwards

    # b) regularization
    if args.reconstruction_option_tv:  # + derivative of TV
        d = d - args.tau_pixel * args.beta_pixel * derivativeTV(dorig, pixel_area, args.tv_eps)  # derivative of TV (option 1) (straight forward) # optional improvement: scale alpha with pixel_area

    if args.reconstruction_option_volume:  # + conservation of volume
        d = VconsFast(mask * extShrink(d, args.tau_pixel * args.alpha_pixel, args.lower_bound, args.upper_bound), d1, d2)  # + conservation of volume (does not work) # optional improvement: scale alpha with pixel_area

    # landweber_pixel default reconstruction consits of: extended soft-shrinkage + mask + heuristic for conservation of volume
    d = mask * extShrink(d, args.tau_pixel * args.alpha_pixel, args.lower_bound, args.upper_bound)  # optional improvement: scale alpha with pixel_area
    d = VconsHeur(d.rename(None), d1, d2, mask, args.gamma, args.volume_radius)

    height_field_recon_d.rename_(None)
    height_field_recon_d.data = d.rename(None)
    height_field_recon_d.rename_('height', 'width')

    tbl.log_tensor('Height_Field_Estimated', lambda: height_field_recon_d + args.height_offset)
    tbl.log_mesh('Height_Field_Estimated_Mesh', lambda: height_field_to_mesh(coords, height_field_recon_d + args.height_offset, height_field_exact))

    def slice_figure():
        fig = plt.figure()
        plt.gca().set(aspect=1)
        plt.plot(coords.select('dim', 0).select('height', args.height_field_resolution // 2).cpu().numpy(), height_field_exact.select('height', args.height_field_resolution // 2).cpu().numpy(), '-b')
        plt.plot(coords.select('dim', 0).select('height', args.height_field_resolution // 2).cpu().numpy(), (height_field_recon_d + args.height_offset).detach().select('height', args.height_field_resolution // 2).cpu().numpy(), '--r')
        return fig
    tbl.log_figure("Center_Slice", slice_figure)

    if discrepancy <= args.tau_dis * args.noise_level:
        tqdm.write(" ")
        tqdm.write("Iter {}, dis {:2.4f}, err {:2.4f}, fdis {:2.4f}, fspa {:2.4f}, ftv {:2.4f}".format(str(i).zfill(len(str(args.num_iterations))), discrepancy, err, fdis, fspa, ftv.item()))  # output of last iteration
        return False, simulation_recon

    return True, simulation_recon


@tbl(print_step=10)
def optimization_loop_landweber_wavelet(i, args, optim, coords, iors, light_pos, pixel_area, pixel_area_brightness, printing_volume, mask, height_field_exact,
                                        sns, sns_norm, xfm, ifm, yl, yh, update_scene, compute_normals_at_hit, compute_differential_normals_at_hit, mean_energy=1e-5):
    optim.zero_grad()

    # back to pixel space
    height_field_recon_d = ifm((yl, yh))[0, 0].rename('height', 'width')

    height_field_recon = height_field_recon_d + args.height_offset
    update_scene(height_field_recon)

    simulation_recon = sum(compute_recursive_refraction(iors, (args.photon_map_size, args.photon_map_size), args.max_pixel_radius, compute_normals_at_hit, compute_differential_normals_at_hit,
                                                        compute_point_light_dirs(args.height_offset, iors.numel(), light_pos, coords, num_simul=args.num_simulations, num_inner_simul=args.num_inner_simulations, smoothing=args.splat_smoothing, energy=mean_energy))
                           for i in range(args.num_simulations))

    tbl.log_tensor('Simulation_Estimated', lambda: simulation_recon)
    err = (th.norm(height_field_exact.rename(None) - (height_field_recon_d.detach().rename(None) + args.height_offset), 2) / th.norm(height_field_exact.rename(None), 2)).item()  # relative error
    tbl.log_scalar('Error', lambda: err)

    fdis = args.objective_func(simulation_recon, sns)
    discrepancy = (th.sqrt(2 * fdis) / sns_norm).item()
    tbl.log_scalar('Discrepancy', lambda: discrepancy)

    fspa = args.alpha_pixel * th.norm(height_field_recon_d.rename(None), 1)
    tbl.log_scalar('F_Sparsity_Pixel', lambda: fspa.item())

    fspaw = args.alpha_wavelet * (th.norm(yl.detach(), 1) + th.norm(yh[0].detach(), 1) + th.norm(yh[1].detach(), 1) + th.norm(yh[2].detach(), 1))  # sparsity of wavelet coefficients # optional: scale with pixel_area
    tbl.log_scalar('F_Sparsity_Wavelet', lambda: fspaw.item())

    # iterations: first i is 0; Iter 0 shows values before first reconstruction iteration
    if i <= 10 or i % 10 == 0 or i > args.num_iterations - 10:
        tqdm.write("Iter {}, dis {:2.4f}, err {:2.4f}, fdis {:2.4f}, fspa {:2.4f}, fspaw {:2.4f}".format(str(i).zfill(len(str(args.num_iterations))), discrepancy, err, fdis, fspa, fspaw))

    fdis.backward()

    tbl.log_tensor('YL_grad', lambda: yl.grad.squeeze(0))
    tbl.log_tensor('YH[0]_grad', lambda: yh[0].grad.squeeze(0))
    tbl.log_tensor('YH[1]_grad', lambda: yh[1].grad.squeeze(0))
    tbl.log_tensor('YH[2]_grad', lambda: yh[2].grad.squeeze(0))

    # a) gradient descent
    yl.data -= args.tau_wavelet * yl.grad.data
    yh[0].data -= args.tau_wavelet * yh[0].grad.data
    yh[1].data -= args.tau_wavelet * yh[1].grad.data
    yh[2].data -= args.tau_wavelet * yh[2].grad.data

    # b) regularization
    # p1, p2 correspond to height_field_recon_d and not to wavelet coefficients
    # workaround...
    p1w = -1E4
    p2w = 1E4
    # extended soft-shrinkage (otpional: scale with pixel_area):
    yl.data = extShrink(yl.data, args.tau_wavelet * args.alpha_wavelet, p1w, p2w)
    yh[0].data = extShrink(yh[0].data, args.tau_wavelet * args.alpha_wavelet, p1w, p2w)
    yh[1].data = extShrink(yh[1].data, args.tau_wavelet * args.alpha_wavelet, p1w, p2w)
    yh[2].data = extShrink(yh[2].data, args.tau_wavelet * args.alpha_wavelet, p1w, p2w)

    # compute height_field_recon_d from wavelet coefficients (repetition of code above...)
    height_field_recon_d = ifm((yl.data, (yh[0].data, yh[1].data, yh[2].data)))[0, 0]

    # Veps = 0.05  # max. relative error of printed volume
    d1 = (printing_volume - printing_volume * args.volume_eps) / pixel_area
    d2 = (printing_volume + printing_volume * args.volume_eps) / pixel_area

    # Do some regularization on pixel basis

    # extended soft-shrinkage on pixel basis... (mathematically not straight forward apart from pyhsical bounds p1 and p2 but works)
    height_field_recon_d = mask * extShrink(height_field_recon_d, args.tau_pixel * args.alpha_pixel, args.lower_bound, args.upper_bound)
    height_field_recon_d = VconsHeur(height_field_recon_d.rename(None), d1, d2, mask, args.gamma, args.volume_radius)

    wd = xfm(height_field_recon_d.rename(None)[None, None, :, :])  # wavelet coefficients (not needed in all cases...)
    yl.data, yh[0].data, yh[1].data, yh[2].data = wd[0], wd[1][0], wd[1][1], wd[1][2]  # wavelet coefficients

    tbl.log_tensor('YL', lambda: yl.squeeze(0))
    tbl.log_tensor('YH[0]', lambda: yh[0].squeeze(0))
    tbl.log_tensor('YH[1]', lambda: yh[1].squeeze(0))
    tbl.log_tensor('YH[2]', lambda: yh[2].squeeze(0))

    tbl.log_tensor('Height_Field_Estimated', lambda: height_field_recon_d + args.height_offset)
    tbl.log_mesh('Height_Field_Estimated_Mesh', lambda: height_field_to_mesh(coords, height_field_recon_d + args.height_offset, height_field_exact))

    def slice_figure():
        fig = plt.figure()
        plt.gca().set(aspect=1)
        plt.plot(coords.select('dim', 0).select('height', args.height_field_resolution // 2).cpu().numpy(), height_field_exact.select('height', args.height_field_resolution // 2).cpu().numpy(), '-b')
        plt.plot(coords.select('dim', 0).select('height', args.height_field_resolution // 2).cpu().numpy(), (height_field_recon_d + args.height_offset).detach().select('height', args.height_field_resolution // 2).cpu().numpy(), '--r')
        return fig
    tbl.log_figure("Center_Slice", slice_figure)

    if discrepancy <= args.tau_dis * args.noise_level:
        tqdm.write(" ")
        tqdm.write("Iter {}, dis {:2.4f}, err {:2.4f}, fdis {:2.4f}, fspa {:2.4f}, fspaw {:2.4f}".format(str(i).zfill(len(str(args.num_iterations))), discrepancy, err, fdis, fspa, fspaw))  # information output of last iteration
        return False, simulation_recon

    return True, simulation_recon


@tbl(print_step=1)
def main(args):
    # in case it is missed somewhere in the code
    th.set_default_dtype(args.dtype)
    th.cuda.set_device(args.device)
    th.cuda.init()

    # logging
    tbl.writer = th.utils.tensorboard.SummaryWriter(os.path.join('runs', args.datetime))
    tbl.global_step = 0
    tbl.key = "Shape_From_Caustics"

    tbl.log_text('Parameters', lambda: str(args))

    light_pos = th.from_numpy(np.asarray(args.light_pos)).to(dtype=args.dtype, device=args.device).refine_names('dim')
    iors = fused_silica(th.from_numpy(np.asarray(args.wavelengths)).to(dtype=args.dtype, device=args.device)).refine_names('channel')  # index of reflection (in dependence of wavelength)

    if args.reconstruct:
        coords = th.stack(th.meshgrid(th.linspace(-1, 1, args.height_field_resolution, dtype=args.dtype), th.linspace(-1, 1, args.height_field_resolution, dtype=args.dtype))).flip(0).refine_names('dim', 'height', 'width').to(args.device)
        height_field_exact = generate_height_field(args, coords)

        tbl.log_tensor('Height_Field_Reference', lambda: height_field_exact)
    else:
        # FIXME: hardcoded coords same as for the reconstruction
        coords = th.stack(th.meshgrid(th.linspace(-1, 1, args.height_field_resolution, dtype=args.dtype), th.linspace(-1, 1, args.height_field_resolution, dtype=args.dtype))).flip(0).refine_names('dim', 'height', 'width').to(args.device)

    # create a mesh to represent our scene
    if args.reconstruct:
        scene = create_from_height_field(coords, height_field_exact, get_normal_from_height_field(height_field_exact, (coords[:, 1, 1] - coords[:, 0, 0])), sensor_height=args.screen_position, additional_elements=args.additional_elements)
    else:
        scene = create_from_height_field(None, None, None, sensor_height=args.screen_position, additional_elements=args.additional_elements)

    def get_normal_at_hit(oi, ti, uv):
        r = scene.prepare_hit_information(oi, ti, uv, requested_params=['normal', 'geometric_normal'])
        return r['geometric_normal'].refine_names('sample', 'dim'), r['normal'].refine_names('sample', 'dim')

    # mask of height field (Which pixels does not have the value hOffset? + local inaccuracy)
    if args.reconstruct and args.read_gt is None:
        height_field_exact_d = height_field_exact - args.height_offset  # d_exact is the height of the printing on top of the glass block
        mask = (th.abs(height_field_exact_d) >= args.mask_zero_eps)
        # Make the mask bigger as the 3D print may be inaccurate:
        n0 = mask.size()[0]
        n1 = mask.size()[1]
        # maskNP = 0.05  # percent of neighbor pixels to increase the mask
        n = int(max(1, np.floor(args.mask_np * max(n0, n1))))  # number of neighbor pixels
        maskNew = mask.clone()  # clone the mask instead of new reference
        for ni in range(n, n0 - n + 1):  # avoid min and max index
            for nj in range(n, n1 - n + 1):  # avoid min and max index
                if mask[ni, nj] == 1:
                    maskNew[ni - n:ni + n + 1, nj - n:nj + n + 1] = 2
        mask = maskNew

        tbl.log_tensor('Mask', lambda: mask)

    if args.read_gt is None:
        simulation_exact = sum(compute_recursive_refraction(iors, (args.photon_map_size_reference, args.photon_map_size_reference), args.max_pixel_radius, get_normal_at_hit, scene.differential_normal,
                                                            compute_point_light_dirs(args.height_offset, iors.numel(), light_pos, coords,
                                                                                     num_simul=args.num_simulations_reference, num_inner_simul=args.num_inner_simulations_reference, smoothing=args.splat_smoothing_reference))
                               for i in tqdm(range(args.num_simulations_reference)))

        # debug plot
        tbl.log_tensor('Reference_Simulation', lambda: simulation_exact)
        if not args.reconstruct:
            th.save({
                    'Parameters': args,
                    'Reference_Simulation': simulation_exact.rename(None),
                    }, os.path.join('savestates', '{}.pts'.format(args.datetime)))

    if args.reconstruct:
        if args.read_gt is None:
            simulation_exact_recon = sum(compute_recursive_refraction(iors, (args.photon_map_size_reference, args.photon_map_size_reference), args.max_pixel_radius, get_normal_at_hit, scene.differential_normal,
                                                                      compute_point_light_dirs(args.height_offset, iors.numel(), light_pos, coords,
                                                                                               num_simul=args.num_simulations_reference, num_inner_simul=args.num_inner_simulations_reference, smoothing=args.splat_smoothing_reference))
                                         for i in tqdm(range(args.num_simulations_reference)))
            simulation_exact_scaled = F.interpolate(simulation_exact.rename(None).unsqueeze(0), size=(args.photon_map_size, args.photon_map_size), mode='bilinear').squeeze(0).refine_names('channel', 'height', 'width')

            noise_level_sim = th.norm(simulation_exact_recon.rename(None) - simulation_exact_scaled.rename(None), 'fro') / th.norm(simulation_exact_scaled.rename(None), 'fro')
            del simulation_exact_recon

            # Noise level: notation:
            # args.noise_level: desired noise level
            # noise_level_sim: intrinsic noise level of the simulation
            # noise_level_add: noise level to add to intrinsic noise level to reach the desired noise level
            # noise_level_final: noise_level_sim + noise_level_add = args.noise_level (for discrepancy principle) (partly from simulation and partly from added noise) (be careful it is an approximation...)

            print("Desired noise level: ", args.noise_level)
            print("Intrinsic noise level: ", noise_level_sim)
            # Is the desired noise level lower than the intrinsic noise level (warn the user)
            if args.noise_level < noise_level_sim:
                sys.exit("Warning: The desired noise level is lower than the intrinsic noise level of the simulation. Choose a higher noise level, i.e. noise_level. Program is terminated.")

            noise_level_add = args.noise_level - noise_level_sim
            print("Add noise level: ", noise_level_add)

            # add noise to exact simulation
            noiseNormal = th.empty_like(simulation_exact).normal_(mean=0, std=1)  # noise with normal distribution
            noise = noiseNormal / th.norm(noiseNormal.rename(None), 'fro') * th.norm(simulation_exact.rename(None), 'fro') * noise_level_add
            simulation_noise = simulation_exact + noise  # simulation with noise

            tbl.log_tensor('Reference_Simulation_Noise', lambda: simulation_noise)

            simulation_noise_scaled = F.interpolate(simulation_noise.rename(None).unsqueeze(0), size=(args.photon_map_size, args.photon_map_size), mode='bilinear').squeeze(0).refine_names('channel', 'height', 'width')
        else:
            simulation_noise_scaled = th.from_numpy(np.array(Image.open(args.read_gt).resize((args.photon_map_size, args.photon_map_size)))).to(args.dtype).to(args.device) / 255.
            simulation_noise_scaled = simulation_noise_scaled.unsqueeze(0).refine_names('channel', 'height', 'width')
            tbl.log_tensor('Reference_Simulation_Noise', lambda: simulation_noise_scaled)
            simulation_exact = simulation_noise_scaled
            simulation_noise = simulation_noise_scaled

            mask = th.from_numpy(np.array(Image.open(args.mask_image).resize((args.height_field_resolution, args.height_field_resolution)))).to(args.dtype).to(args.device) / 255.
            mask = mask.refine_names('height', 'width')

        if args.a_priori == 'none':
            height_field_recon_d = th.zeros_like(height_field_exact)
        elif args.a_priori == 'known' and args.read_gt is None:
            height_field_recon_d = height_field_exact - args.height_offset

        # Prepare wavelets (if used)
        if args.reconstruction_method == 'landweber_wavelet':
            xfm = DWTForward(J=3, mode='zero', wave=args.wavelet).cuda(args.device)
            ifm = DWTInverse(mode='zero', wave=args.wavelet).cuda(args.device)

            wd = xfm(height_field_recon_d.rename(None)[None, None, :, :])
            yl, yh = wd  # wavelet coefficients

            if args.a_priori == 'none':
                yl.zero_()
                yh[0].zero_()
                yh[1].zero_()
                yh[2].zero_()

        pixel_area = (coords[0][0][1] - coords[0][0][0]) * (coords[1][1][0] - coords[1][0][0])  # area of one pixel of the height field (notation 'a' is used already)
        # area of one pixel on the sensor (i.e. simulation_recon, simulation_exact, ...); in the used experimental set-up the sensor has the same physical size as the glass substrate
        pixel_area_brightness = (th.sqrt(pixel_area) * args.height_field_resolution / args.photon_map_size)**2

        if args.read_gt is None:
            printing_volume = (pixel_area * th.sum(height_field_exact_d)).item()  # exact volume of added glass on the glass block (area*(sum of d))
        else:
            printing_volume = args.deposited_volume
        tbl.log_text("Exact_Printing_Volume", lambda: str(printing_volume))

        if args.reconstruction_method == 'landweber_pixel' or args.reconstruction_method == 'baseline':
            height_field_recon_d.requires_grad = True
            model = [height_field_recon_d]
        elif args.reconstruction_method == 'landweber_wavelet':
            yl.requires_grad = True
            yh[0].requires_grad = True
            yh[1].requires_grad = True
            yh[2].requires_grad = True
            model = [yl, yh[0], yh[1], yh[2]]

        optim = th.optim.SGD(model, lr=args.tau_pixel)  # learning rate has no influence as optim.step() is not used

        try:
            for i in tqdm(range(1, args.num_iterations + 1)):  # reconstruction iteration
                tbl.global_step = i

                if args.reconstruction_method == 'baseline':
                    cont, simulation_recon = optimization_loop_baseline(i, args, optim, coords, iors, light_pos, height_field_exact, height_field_recon_d, simulation_noise_scaled, th.norm(simulation_noise_scaled.rename(None), 'fro'),
                                                                        lambda hr: scene['height_field_mesh'].update_from_height_field(coords, hr, get_normal_from_height_field(hr, (coords[:, 1, 1] - coords[:, 0, 0]))),
                                                                        get_normal_at_hit, scene.differential_normal, mean_energy=args.energy)
                elif args.reconstruction_method == 'landweber_pixel':
                    cont, simulation_recon = optimization_loop_landweber_pixel(i, args, optim, coords, iors, light_pos, pixel_area, pixel_area_brightness, printing_volume, mask, height_field_exact,
                                                                               height_field_recon_d, simulation_noise_scaled, th.norm(simulation_noise_scaled.rename(None), 'fro'),
                                                                               lambda hr: scene['height_field_mesh'].update_from_height_field(coords, hr, get_normal_from_height_field(hr, (coords[:, 1, 1] - coords[:, 0, 0]))),
                                                                               get_normal_at_hit, scene.differential_normal, mean_energy=args.energy)  # optional: scale th.norm with pixel_area_brightness(?)
                elif args.reconstruction_method == 'landweber_wavelet':
                    cont, simulation_recon = optimization_loop_landweber_wavelet(i, args, optim, coords, iors, light_pos, pixel_area, pixel_area_brightness, printing_volume, mask, height_field_exact, simulation_noise_scaled,
                                                                                 th.norm(simulation_noise_scaled.rename(None), 'fro'), xfm, ifm, yl, yh,
                                                                                 lambda hr: scene['height_field_mesh'].update_from_height_field(coords, hr, get_normal_from_height_field(hr, (coords[:, 1, 1] - coords[:, 0, 0]))),
                                                                                 get_normal_at_hit, scene.differential_normal, mean_energy=args.energy)  # optional: scale th.norm with pixel_area_brightness(?)

                if not cont:
                    break
        finally:
            tqdm.write("Stopped after {} iterations based on {}".format(tbl.global_step, "stopping criterion" if tbl.global_step != args.num_iterations else "iteration count"))
            th.save({
                    'Iteration': tbl.global_step,
                    'Parameters': args,
                    'Height_Field_Exact': height_field_exact.rename(None),
                    'Reference_Simulation': simulation_exact.rename(None),
                    'Reference_Simulation_Noise': simulation_noise.rename(None),
                    'Simulation_Recon': simulation_recon.rename(None),
                    'Optimization_Parameters': [m.rename(None) for m in model],
                    }, os.path.join('savestates', '{}.pts'.format(args.datetime)))

            if args.reconstruction_method == 'landweber_wavelet':
                height_field_recon_d = ifm((yl, yh))[0, 0].rename('height', 'width')
            post_optimization_plot(args, coords.detach().cpu().numpy(), height_field_exact.detach().cpu().numpy(), (height_field_recon_d + args.height_offset).detach().cpu().numpy(), simulation_noise_scaled.detach().cpu().numpy())
            tbl.log_tensor('Height_Field_Estimated', lambda: height_field_recon_d + args.height_offset)


if __name__ == "__main__":
    for args in get_argument_set():
        main(args)
