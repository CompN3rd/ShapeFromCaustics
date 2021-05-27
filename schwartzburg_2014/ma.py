import sys
sys.path.append('../ThirdParty/PyMongeAmpere')
import MongeAmpere as ma

sys.path.append('../ThirdParty/cgal-python')
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
from CGAL.CGAL_Interpolation import natural_neighbor_coordinates_2, linear_interpolation, Data_access_double_2

sys.path.append('.')

import torch as th
import torch.nn.functional as F

from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import pairwise_distances_argmin
import scipy.optimize as opt
from datetime import datetime


def draw_laguerre_cells(dens, Y, w):
    E = dens.restricted_laguerre_edges(Y, w)
    nan = float('nan')
    N = E.shape[0]
    x = np.zeros(3 * N)
    y = np.zeros(3 * N)
    a = np.array(range(0, N))
    x[3 * a] = E[:, 0]
    x[3 * a + 1] = E[:, 2]
    x[3 * a + 2] = nan
    y[3 * a] = E[:, 1]
    y[3 * a + 1] = E[:, 3]
    y[3 * a + 2] = nan
    plt.plot(x, y, color=[1, 0, 0], linewidth=1, aa=True)


def single_lloyd(dens0, dens1, N=20000, num_smooth=100, verbose=True):
    sites0 = dens0.random_sampling(N)

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')
        plt.cla()
        draw_laguerre_cells(dens0, sites0, np.zeros(sites0.shape[0]))
        plt.axis([-1, 1, -1, 1])
        plt.pause(0.1)

    for i in trange(num_smooth):
        opt_sites, m = dens0.lloyd(sites0)
        sites0[m > 0] = opt_sites[m > 0]

        if verbose:
            plt.cla()
            draw_laguerre_cells(dens0, sites0, np.zeros(sites0.shape[0]))
            plt.axis([-1, 1, -1, 1])
            plt.pause(0.1)

    if verbose:
        plt.pause(5)

    sites1 = sites0.copy()

    if verbose:
        plt.cla()
        draw_laguerre_cells(dens1, sites1, np.zeros(sites1.shape[0]))
        plt.axis([-1, 1, -1, 1])
        plt.pause(0.1)

    for i in trange(num_smooth):
        opt_sites, m = dens1.lloyd(sites1)
        sites1[m > 0] = opt_sites[m > 0]

        if verbose:
            plt.cla()
            draw_laguerre_cells(dens1, sites1, np.zeros(sites1.shape[0]))
            plt.axis([-1, 1, -1, 1])
            plt.pause(0.1)

    if verbose:
        plt.pause(5)
        plt.close()

    return sites0, sites1


def make_multiscale_from_dens(dens, N=2**15, divisor=4, num_smooth=100, verbose=True):
    sites = [dens.random_sampling(N)]

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')

    for i in trange(int(np.floor(np.log(N) / np.log(divisor))) - 1):
        # for source density
        # smooth with lloyds sampling
        if verbose:
            plt.cla()
            draw_laguerre_cells(dens, sites[-1], np.zeros(sites[-1].shape[0]))
            plt.axis([-1, 1, -1, 1])
            plt.pause(.01)

        for i in trange(num_smooth):
            opt_sites, m = dens.lloyd(sites[-1])
            sites[-1][m > 0] = opt_sites[m > 0]

            if verbose:
                plt.cla()
                draw_laguerre_cells(dens, sites[-1], np.zeros(sites[-1].shape[0]))
                plt.axis([-1, 1, -1, 1])
                plt.pause(.01)

        if verbose:
            plt.pause(5)

        # subsample the sites for next iteration
        sites.append(sites[-1].copy()[np.random.choice(sites[-1].shape[0], size=sites[-1].shape[0] // divisor, replace=False)])

    if verbose:
        plt.close()

    return sites


def make_multiscale_hierarchy(dens0, dens1, N=20000, divisor=4, num_smooth=75, verbose=True):
    # build a hierarchy of sites
    sites0 = [dens0.random_sampling(N)]
    sites1 = []

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')

    for i in trange(int(np.floor(np.log(N) / np.log(divisor))) - 1):
        # for source density
        # smooth with lloyds sampling
        if verbose:
            plt.cla()
            draw_laguerre_cells(dens0, sites0[-1], np.zeros(sites0[-1].shape[0]))
            plt.axis([-1, 1, -1, 1])
            plt.pause(.01)

        for i in trange(num_smooth):
            opt_sites, m = dens0.lloyd(sites0[-1])
            sites0[-1][m > 0] = opt_sites[m > 0]

            if verbose:
                plt.cla()
                draw_laguerre_cells(dens0, sites0[-1], np.zeros(sites0[-1].shape[0]))
                plt.axis([-1, 1, -1, 1])
                plt.pause(.01)

        if verbose:
            plt.pause(5)

        # for target density
        # start with final source sites
        sites1.append(sites0[-1].copy())
        if verbose:
            plt.cla()
            draw_laguerre_cells(dens1, sites1[-1], np.zeros(sites1[-1].shape[0]))
            plt.axis([-1, 1, -1, 1])
            plt.pause(.01)

        # and smooth it, too
        for i in trange(num_smooth):
            opt_sites, m = dens1.lloyd(sites1[-1])
            sites1[-1][m > 0] = opt_sites[m > 0]

            if verbose:
                plt.cla()
                draw_laguerre_cells(dens1, sites1[-1], np.zeros(sites1[-1].shape[0]))
                plt.axis([-1, 1, -1, 1])
                plt.pause(.01)

        if verbose:
            plt.pause(5)

        # subsample the source sites for next iteration
        sites0.append(sites0[-1].copy()[np.random.choice(sites0[-1].shape[0], size=sites0[-1].shape[0] // divisor, replace=False)])

    # make target sites same length
    sites1.append(sites0[-1].copy())

    if verbose:
        plt.cla()
        draw_laguerre_cells(dens1, sites1[-1], np.zeros(sites1[-1].shape[0]))
        plt.axis([-1, 1, -1, 1])
        plt.pause(.01)

    # and smooth it, too
    for i in trange(num_smooth):
        opt_sites, m = dens1.lloyd(sites1[-1])
        sites1[-1][m > 0] = opt_sites[m > 0]

        if verbose:
            plt.cla()
            draw_laguerre_cells(dens1, sites1[-1], np.zeros(sites1[-1].shape[0]))
            plt.axis([-1, 1, -1, 1])
            plt.pause(.01)

    if verbose:
        plt.pause(5)
        plt.close()

    return sites0, sites1


def natural_neighbor_interpolate(sample_points, data, query_points, verbose=True):
    # triangulate data points
    triang = Delaunay_triangulation_2()
    triang.insert([Point_2(p[0], p[1]) for p in sample_points])

    # border points
    min_border = query_points.min(0)
    max_border = query_points.max(0)
    triang.insert([Point_2(min_border[0], min_border[1]), Point_2(max_border[0], min_border[1]), Point_2(max_border[0], max_border[1]), Point_2(min_border[0], max_border[1])])

    # make data access wrapper thingy for linear_interpolate
    dat = [Data_access_double_2() for d in range(data.shape[1])]

    for p, c in tqdm(zip(sample_points, data)):
        for d, cd in zip(dat, c):
            d.set(Point_2(p[0], p[1]), cd)

    # border points
    for d, mib, mab in zip(dat, min_border, max_border):
        d.set(Point_2(min_border[0], min_border[1]), mib)
        d.set(Point_2(max_border[0], min_border[1]), mab if d is dat[0] else mib)
        d.set(Point_2(max_border[0], max_border[1]), mab)
        d.set(Point_2(min_border[0], max_border[1]), mib if d is dat[0] else mab)

    output = []
    for q in tqdm(query_points):
        coords = []
        norm = natural_neighbor_coordinates_2(triang, Point_2(q[0], q[1]), coords)[0]
        output.append([linear_interpolation(coords, norm, d) for d in dat])

    return np.asarray(output)


# geometry calculation (numpy)
def compute_incident_dir(x, light_pos):
    diff = x - light_pos[None, :]
    return diff / np.linalg.norm(diff, axis=1, keepdims=True)


def compute_refracted_dir(incident_dir, normal, eta_it):
    cosi = (-incident_dir * normal).sum(axis=1, keepdims=True)
    k = 1 + eta_it**2 * (cosi**2 - 1)
    return eta_it * incident_dir + (eta_it * cosi - np.sqrt(k)) * normal


def compute_refracted_intersection(x, normal, incident_dir, receiver_plane, eta):
    d_r = compute_refracted_dir(incident_dir, normal, 1 / eta)

    # assumes system is aligned in z-direction
    return x + ((receiver_plane - x[:, 2:]) / d_r[:, 2:]) * d_r


def otm_interpolation(x, normal, voronoi_points, voronoi_data, light_pos, receiver_plane, eta, verbose=False):
    d_i = compute_incident_dir(x, light_pos)

    isect = compute_refracted_intersection(x, normal, d_i, receiver_plane, eta)[:, :2]

    # necessary for interpolation to work
    # assert((isect >= -1).all() and (isect <= 1).all())
    # isect = np.clip(isect, -1 + 1e-7, 1 - 1e-7)

    target_points = np.zeros_like(x)
    target_points[:, :2] = natural_neighbor_interpolate(voronoi_points, voronoi_data, isect, verbose=verbose)
    target_points[:, 2] = receiver_plane

    return target_points


def fresnel_mapping(x, d_i, target_points, eta):
    # definition from schwartzburg eta = eta'
    diff = target_points - x
    d_t = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    return (d_i - eta * d_t) / np.linalg.norm(d_i - eta * d_t, axis=1, keepdims=True)


# -------------------------------------------------------------------------
# target optimization
# -------------------------------------------------------------------------
# optimization terms (pytorch)
def compute_normals(x):
    # graph looks like this
    #       (n-1, n-1)
    # *--*--*
    # |\ |\ |
    # | \| \|
    # *--*--*
    # |\ |\ |
    # | \| \|
    # *--*--*
    # (0, 0)
    # so each non-border vertex has 6 neighbors
    # compute necessary differences:

    diff_x_raw = x[:, 1:] - x[:, :-1]  # right - left
    norm = th.norm(diff_x_raw, p=2, dim=-1)
    mask = norm.gt(0)
    diff_x = th.zeros_like(diff_x_raw)
    diff_x[mask] = diff_x_raw[mask] / norm[mask].unsqueeze(-1)

    diff_y_raw = x[1:, :] - x[:-1, :]  # bottom - top
    norm = th.norm(diff_y_raw, p=2, dim=-1)
    mask = norm.gt(0)
    diff_y = th.zeros_like(diff_y_raw)
    diff_y[mask] = diff_y_raw[mask] / norm[mask].unsqueeze(-1)

    diff_diag_raw = x[1:, 1:] - x[:-1, :-1]  # bottom right - top left
    norm = th.norm(diff_diag_raw, p=2, dim=-1)
    mask = norm.gt(0)
    diff_diag = th.zeros_like(diff_diag_raw)
    diff_diag[mask] = diff_diag_raw[mask] / norm[mask].unsqueeze(-1)

    # bottom right triangle lower
    normal = F.pad(th.cross(diff_y[:, :-1], diff_diag), (0, 0, 0, 1, 0, 1))
    # bottom right triangle upper
    normal += F.pad(th.cross(diff_diag, diff_x[:-1]), (0, 0, 0, 1, 0, 1))

    # top right triangle
    normal += F.pad(th.cross(diff_x[1:], -diff_y[:, :-1]), (0, 0, 0, 1, 1, 0))

    # top left triangle upper
    normal += F.pad(th.cross(-diff_y[:, 1:], -diff_diag), (0, 0, 1, 0, 1, 0))
    # top left triangle lower
    normal += F.pad(th.cross(-diff_diag, -diff_x[1:]), (0, 0, 1, 0, 1, 0))

    # bottom left triangle
    normal += F.pad(th.cross(-diff_x[:-1], diff_y[:, 1:]), (0, 0, 1, 0, 0, 1))

    # renormalize
    norm = th.norm(normal, p=2, dim=-1)
    mask = norm.gt(0)

    normalized_normal = th.zeros_like(x)
    normalized_normal[mask] = normal[mask] / norm[mask].unsqueeze(1)

    # plt.figure(figsize=(10, 10), facecolor='white')
    # plt.imshow(0.5 * normalized_normal.detach().cpu().numpy() + 0.5)
    # plt.show()

    return normalized_normal


def E_int(current_normals, target_normals):
    return ((current_normals - target_normals)**2).sum()


def E_dir(x, x_s, d_i):
    # project x onto line (x_s, d_i)
    proj = ((x - x_s) * d_i).sum(1, True) * d_i + x_s

    return ((x - proj)**2).sum()

# can't make E_flux auto-differentiable, so I leave it out for now
# def E_flux(weight, )


def E_reg(x):
    # norm of laplacian of vector positions
    # graph looks like this
    # *--*--*
    # |\ |\ |
    # | \| \|
    # *--*--*
    # |\ |\ |
    # | \| \|
    # *--*--*
    # so each non-border vertex has 6 neighbors
    # pad the border with replicates
    x_pad = F.pad(x.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0).permute(1, 2, 0)

    # starting at bottom right
    lx = 6 * x_pad[1:-1, 1:-1] - x_pad[2:, 2:] - x_pad[1:-1, 2:] - x_pad[:-2, 1:-1] - x_pad[:-2, :-2] - x_pad[1:-1, :-2] - x_pad[2:, 1:-1]

    return (lx**2).sum()


def E_bar(x, receiver_plane, d_th):
    return (F.relu(-th.log(1 - (receiver_plane - x[:, 2]) + d_th))**2).sum()


def normal_integration(x_numpy, n_r, x_s, d_i, size, receiver_plane, verbose=True):
    x = th.from_numpy(x_numpy).cuda().requires_grad_()
    x_old = x.detach().clone()
    optim = th.optim.LBFGS([x], line_search_fn='strong_wolfe')

    n_r_cuda = th.from_numpy(n_r).cuda()
    d_i_cuda = th.from_numpy(d_i).cuda()
    x_s_cuda = th.from_numpy(x_s).cuda()

    losses = [0]
    normal_losses = [0]
    smooth_losses = [0]
    barrier_losses = [0]
    direction_losses = [0]

    def closure():
        x_flat = x.view(-1, 3)
        x_grid = x.view(*size, 3)
        normal_loss = 1.0 * E_int(compute_normals(x_grid).view(-1, 3), n_r_cuda)
        barrier_loss = 1.0 * E_bar(x_flat, receiver_plane, -0.1 - 4e-3)
        direction_loss = 5e3 * E_dir(x_flat, x_s_cuda, d_i_cuda)
        smooth_loss = 5e3 * E_reg(x_grid)

        loss = normal_loss + smooth_loss + barrier_loss + direction_loss

        normal_losses[-1] = normal_loss.item()
        smooth_losses[-1] = smooth_loss.item()
        barrier_losses[-1] = barrier_loss.item()
        direction_losses[-1] = direction_loss.item()
        losses[-1] = loss.item()

        optim.zero_grad()
        loss.backward()
        return loss

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')

    while len(losses) <= 2 or abs(losses[-3] - losses[-2]) >= 1e-5 or th.norm(x.detach() - x_old) >= 1e-5:
        optim.step(closure)

        if verbose:
            plt.cla()
            plt.plot(np.arange(1, len(losses) + 1), losses, label='loss')
            plt.plot(np.arange(1, len(losses) + 1), normal_losses, label='normal loss')
            plt.plot(np.arange(1, len(losses) + 1), smooth_losses, label='smooth loss')
            plt.plot(np.arange(1, len(losses) + 1), barrier_losses, label='barrier loss')
            plt.plot(np.arange(1, len(losses) + 1), direction_losses, label='direction_loss')
            plt.gca().legend()
            plt.pause(.01)

        losses.append(0)
        normal_losses.append(0)
        smooth_losses.append(0)
        barrier_losses.append(0)
        direction_losses.append(0)
        x_old = x.detach().clone()

    if verbose:
        plt.pause(5)
        plt.close()

    return x.detach().cpu().numpy(), losses[-2]


def target_optimization(x_init, voronoi_points, voronoi_data, light_pos, receiver_plane, eta, size, eps=1e-2, verbose=True):
    x = x_init.copy()
    n_init = compute_normals(th.from_numpy(x_init.reshape(*size, 3))).reshape(-1, 3).numpy()
    x_old = np.zeros_like(x)
    old_loss = np.inf

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')
        plt.imshow(0.5 * n_init.reshape(*size, 3) + 0.5)
        plt.pause(5)
        plt.close()

    x_r = otm_interpolation(x, n_init, voronoi_points, voronoi_data, light_pos, receiver_plane, eta, verbose=verbose)

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')
        # plt.scatter(x_init[:, 0], x_init[:, 1], s=0.1)
        plt.scatter(x_r[:, 0], x_r[:, 1], s=0.1)
        plt.plot([x_init[0, 0], x_r[0, 0]], [x_init[0, 1], x_r[0, 1]])
        plt.plot([x_init[size[1] - 1, 0], x_r[size[1] - 1, 0]], [x_init[size[1] - 1, 1], x_r[size[1] - 1, 1]])
        plt.plot([x_init[(size[0] - 1) * size[1], 0], x_r[(size[0] - 1) * size[1], 0]], [x_init[(size[0] - 1) * size[1], 1], x_r[(size[0] - 1) * size[1], 1]])
        plt.plot([x_init[(size[0] - 1) * size[1] + size[1] - 1, 0], x_r[(size[0] - 1) * size[1] + size[1] - 1, 0]], [x_init[(size[0] - 1) * size[1] + size[1] - 1, 1], x_r[(size[0] - 1) * size[1] + size[1] - 1, 1]])
        plt.axis([-1, 1, -1, 1])
        plt.pause(5)
        plt.close()

    conv_norm = np.linalg.norm(x - x_old)
    while conv_norm > eps:
        print("Outer Iteration |x_k+1 - x_k| = {}".format(conv_norm))

        d_i = compute_incident_dir(x, light_pos)
        n_r = fresnel_mapping(x, d_i, x_r, eta)

        if verbose:
            plt.figure(figsize=(10, 10), facecolor='white')
            plt.subplot(121)
            plt.imshow(0.5 * n_r.reshape(*size, 3) + 0.5)
            plt.subplot(122)
            current_normals = compute_normals(th.from_numpy(x).view(*size, 3))
            plt.imshow(0.5 * current_normals.numpy() + 0.5)
            plt.pause(5)
            plt.close()

            plt.figure(figsize=(10, 10), facecolor='white')
            x_inter = compute_refracted_intersection(x, n_r, d_i, receiver_plane, eta)
            plt.scatter(x_inter[:, 0], x_inter[:, 1], s=0.1)
            plt.axis([-1, 1, -1, 1])
            plt.pause(5)
            plt.close()

        x_old = x.copy()
        x, loss = normal_integration(x.astype(np.float32), n_r.astype(np.float32), x_init.astype(np.float32), d_i.astype(np.float32), size, receiver_plane, verbose=verbose)
        x = x.reshape(-1, 3).astype(x_old.dtype)

        if np.isnan(loss) or loss >= old_loss * 1e3:
            print("Optimization diverged, reverting old state")
            x = x_old

        old_loss = loss
        conv_norm = np.linalg.norm(x - x_old)

    return x


def write_obj(filepath, target, size, flipY=True):
    target_normals = compute_normals(th.from_numpy(target).reshape(*size, 3)).reshape(-1, 3).numpy()

    with open(filepath, 'w') as f:
        f.write('# OBJ file\n')
        f.write('o Substrate\n')
        for v in tqdm(target, desc='Vertices'):
            f.write('v {} {} {}\n'.format(v[0], -v[1] if flipY else v[1], v[2]))

        for vn in tqdm(target_normals, desc='Vertex Normals'):
            f.write('vn {} {} {}\n'.format(vn[0], -vn[1] if flipY else vn[1], vn[2]))

        for vt in tqdm(target, desc='Vertex Texcoords'):
            f.write('vt {} {}\n'.format(vt[0], -vt[1] if flipY else vt[1]))

        # (i, j)
        #    *--*
        #    |\ |
        #    | \|
        #    *--*
        #     (i+1,j+1)
        formatstr = 'f {0}/{0}/{0} {2}/{2}/{2} {1}/{1}/{1}\n' if flipY else 'f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n'
        for i in trange(size[0] - 1, desc='Indices', leave=True):
            for j in range(size[1] - 1):
                # ONE indexed!!
                # ccw
                # upper triangle
                f.write(formatstr.format(np.ravel_multi_index((i, j), size) + 1, np.ravel_multi_index((i + 1, j + 1), size) + 1, np.ravel_multi_index((i, j + 1), size) + 1))
                # lower triangle
                f.write(formatstr.format(np.ravel_multi_index((i, j), size) + 1, np.ravel_multi_index((i + 1, j), size) + 1, np.ravel_multi_index((i + 1, j + 1), size) + 1))

                # cw
                # f.write('f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n'.format(np.ravel_multi_index((i, j), size) + 1, np.ravel_multi_index((i, j + 1), size) + 1, np.ravel_multi_index((i + 1, j + 1), size) + 1))
                # f.write('f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}\n'.format(np.ravel_multi_index((i, j), size) + 1, np.ravel_multi_index((i + 1, j + 1), size) + 1, np.ravel_multi_index((i + 1, j), size) + 1))


def optimize_power_diagram(dens0, dens1, sites, verbose=True):
    # init as voronoi diagram
    weights = np.zeros(sites[-1].shape[0])
    save_weights = []

    if verbose:
        plt.figure(figsize=(10, 10), facecolor='white')

    for s, s_next in tqdm(zip(reversed(sites), reversed([None] + sites[:-1])), desc='power diagram optimization'):
        nu = dens0.moments(s)[0]

        if verbose:
            def cb(cur_weights):
                plt.cla()
                draw_laguerre_cells(dens1, s, cur_weights)
                plt.axis([-1, 1, -1, 1])
                plt.pause(.01)

        class gradient_helper:
            def objective(self, x):
                self.x = x
                f, m, g, h = dens1.kantorovich(s, nu, x)

                self.grad = g
                return f

            def gradient(self, x):
                if not np.array_equal(self.x, x):
                    _ = self.objective(x)

                return self.grad

        gh = gradient_helper()

        # multiscale lbfgs optimization
        res = opt.minimize(gh.objective, weights, method='L-BFGS-B', jac=gh.gradient, options={'disp': True}, callback=cb if verbose else None)

        if not res.success:
            print("Optimization unsuccessful: {}".format(res.message))

        save_weights.append(res.x.copy())

        # initialize next weights based on nearest neighbor
        if s_next is not None:
            weights = np.zeros(s_next.shape[0])
            min_neighbor_indices = pairwise_distances_argmin(s_next, s, metric='sqeuclidean')
            weights = res.x.copy()[min_neighbor_indices]

    if verbose:
        plt.pause(5)
        plt.close()

    return save_weights


if __name__ == '__main__':
    verbose = False
    start_time = datetime.now()
    img0 = np.array(Image.open('schwartzburg_2014/source.png').convert('L').resize((256, 256), resample=Image.BICUBIC), dtype=float)[::-1]
    img1 = np.array(Image.open('schwartzburg_2014/target.png').convert('L').resize(img0.shape, resample=Image.BICUBIC), dtype=float)[::-1]

    # add noise up to 5% to img1
    # noise_level_add = 0.0463
    # noiseNormal = th.empty_like(simulation_exact).normal_(mean=0, std=1)  # noise with normal distribution
    # noise = noiseNormal / th.norm(noiseNormal.rename(None), 'fro') * th.norm(simulation_exact.rename(None), 'fro') * noise_level_add
    # simulation_noise = simulation_exact + noise  # simulation with noise
    # img1 = F.interpolate(simulation_noise.unsqueeze(0), size=(256, 256), mode='bilinear')[0, 0].cpu().numpy().astype(np.float64)

    dummy_dens = ma.Density_2.from_image(img0)
    dens0 = ma.Density_2.from_image(img0 / dummy_dens.mass())
    dummy_dens = ma.Density_2.from_image(img1)
    dens1 = ma.Density_2.from_image(img1 / dummy_dens.mass())

    if not path.exists('schwartzburg_2014/source.npy'):
        sites = make_multiscale_from_dens(dens0, verbose=verbose)
        # save the sites
        np.save('schwartzburg_2014/source.npy', sites, allow_pickle=True)

    sites = list(np.load('schwartzburg_2014/source.npy', allow_pickle=True))

    # multiscale lbfgs optimization
    if not path.exists('schwartzburg_2014/opt_weights.npy'):
        save_weights = optimize_power_diagram(dens0, dens1, sites, verbose=verbose)
        np.save('schwartzburg_2014/opt_weights.npy', save_weights, allow_pickle=True)

    weights = list(np.load('schwartzburg_2014/opt_weights.npy', allow_pickle=True))

    if not path.exists('schwartzburg_2014/opt_target.npy'):
        # plt.figure(figsize=(10, 10), facecolor='white')
        # plt.cla()
        # draw_laguerre_cells(dens1, sites[0], weights[-1])
        # plt.axis([-1, 1, -1, 1])
        # plt.show()

        # use the highest resolution
        sites = sites[0]
        weights = weights[-1]

        # get the power diagram centroids
        moments = dens1.moments(sites, weights)
        centroids = moments[1] / moments[0][:, None]

        # geometric setup
        light_pos = np.array([0., 0., 10.])
        receiver_plane = -5e-3

        # initial values
        initial_positions = np.asarray(np.meshgrid(np.linspace(-1, 1, 287), np.linspace(-1, 1, 287))).transpose(1, 2, 0)[::-1].reshape(-1, 2)
        initial_positions = np.concatenate((initial_positions, np.full((initial_positions.shape[0], 1), 0.1)), axis=1)

        # TODO: always check ior 1.457 = 0.633µm
        optimized_target = target_optimization(initial_positions, sites, centroids, light_pos, receiver_plane, 1.457 / 1, (287, 287), verbose=verbose)

        np.save('schwartzburg_2014/opt_target.npy', optimized_target.reshape(287, 287, 3), allow_pickle=True)

    target = np.load('schwartzburg_2014/opt_target.npy', allow_pickle=True)

    if not path.exists('schwartzburg_2014/opt_target.obj'):
        write_obj('schwartzburg_2014/opt_target.obj', target.reshape(-1, 3), (287, 287))

    print("Script execution time: {}".format(datetime.now() - start_time))
