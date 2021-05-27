import argparse
import subprocess
import getpass
import torch as th
from itertools import product
import socket
from datetime import datetime
import sys
import os


def get_datetime_identifier():
    current_time = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    host_name = socket.gethostname()
    return '{}_{}'.format(current_time, host_name)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def l2_sq(x, y):
    return 0.5 * ((x - y)**2).sum()


def l1(x, y):
    return th.abs(x - y).sum()


def get_argument_set():
    parser = argparse.ArgumentParser(description='Compute shape from caustics')
    # technical
    parser.add_argument('--gpu', type=int, default=0, help='The number of the gpu to use')
    parser.add_argument('--dtype', choices=[th.float16, th.float32, th.float64], default=th.float32, help='The bit depth of tensors to use')

    # simulation
    parser.add_argument('--height_field_option', choices=['gaussian', 'image', 'gaussian_damage', 'gaussian_two', 'print_lines', 'oblique_lines'], default='print_lines', help='height field functions')
    parser.add_argument('--height_offset', type=float, default=0.1, nargs='*', help='height of the glass substrate')
    parser.add_argument('--screen_position', type=float, default=-0.05, nargs='*', help='z position of the receiver screen')  # save: -5e-3; -4e-3 (OK) (small spot in the middle); -3e-3 too close
    parser.add_argument('--num_simulations_reference', type=int, default=16, nargs='*', help='number of simulation iterations for reference generation')
    parser.add_argument('--num_inner_simulations_reference', type=int, default=128, nargs='*', help='number of simulation iterations for reference generation')
    parser.add_argument('--num_simulations', type=int, default=16, nargs='*', help='number of simulation iterations for each reconstruction iteration')
    parser.add_argument('--num_inner_simulations', type=int, default=32, nargs='*', help='number of parallel simulation iterations for each reconstruction iteration')
    parser.add_argument('--height_field_resolution', type=int, default=128, nargs='*', help='resolution (square) for the height field')
    parser.add_argument('--photon_map_size_reference', type=int, default=512, nargs='*', help='resolution (square) for the photon map during reference generation')
    parser.add_argument('--photon_map_size', type=int, default=512, nargs='*', help='resolution (square) for the photon map for each reconstruction iteration')
    parser.add_argument('--splat_smoothing_reference', type=float, default=500, nargs='*', help='photon smoothing parameter during reference generation')
    parser.add_argument('--splat_smoothing', type=float, default=250, nargs='*', help='photon smoothing parameter for each reconstruction iteration')
    parser.add_argument('--max_pixel_radius', type=int, default=20, nargs='*', help='maximum splatting radius in pixels after which photons are cut off')
    parser.add_argument('--light_pos', type=float, nargs=3, default=[0, 0, 10], help='position of point light')
    # e. g. W3: [0.475, 0.5625, 0.65], W3L: --wavelengths 0.633 1.152 3.392 (typical wavelengths of HeNe laser), W3W: --wavelengths 0.21 3 6.7 (using full range of silica)
    parser.add_argument('--wavelengths', type=float, nargs='*', default=[0.633], help='wavelengths to be considered')
    # e.g. bottom_lens.obj, top_lens.obj
    parser.add_argument('--additional_elements', nargs='*', help='additional elements to be placed in the scene (currently with the same material as our height field)')

    # no reconstruction, just simulate the element passed here
    parser.add_argument('--reconstruct', type=str2bool, default=True, help='whether to just simulate and or reconstruct as well')

    # no simulation, read gt from file
    parser.add_argument('--read_gt', default=None, help='whether to simulate the ground truth or read it from a file')
    parser.add_argument('--mask_image', default=None, help='the mask image')
    parser.add_argument('--deposited_volume', type=float, default=0.0669, help='the deposited material volume in our units (i.e. 2.5cm <-> 1 unit)')
    parser.add_argument('--energy', default=1e-4, help='the energy of the light source')

    # reconstruction
    parser.add_argument('--a_priori', choices=['none', 'known'], default='none', help='a priori knowledge about the geometry (influences initial value)')
    parser.add_argument('--data_norm', choices=['l1', 'l2_sq'], default='l2_sq', help='utilized data term norm in reconstruction')
    parser.add_argument('--noise_level', type=float, default=0.05, nargs='*', help='relative noise level of perturbed data')
    parser.add_argument('--num_iterations', type=int, default=200, help='number of reconstruction iterations')
    parser.add_argument('--tau_dis', type=float, default=1.1, nargs='*', help='Morozov\'s discrepancy principle; stopping criterion')
    parser.add_argument('--reconstruction_method', choices=['baseline', 'landweber_pixel', 'landweber_wavelet'], default='landweber_pixel', help='reconstruction method')

    parser.add_argument('--reconstruction_option_tv', type=str2bool, default=False, help='reconstruction additionally uses derivative of TV (only in the case of landweber_pixel; experimental option')  # --reconstruction_option_tv true
    parser.add_argument('--beta_pixel', type=float, default=0.01, nargs='*', help='step size of tv is tau * beta')
    parser.add_argument('--tv_eps', type=float, default=None, nargs='*', help='stabilization of total variation')

    parser.add_argument('--reconstruction_option_volume', type=str2bool, default=False, help='reconstruction additionally uses volume constraints (only in the case of landweber_pixel; experimental option; not recommended')

    parser.add_argument('--tau_pixel', type=float, default=1e-1, nargs='*', help='step size of the Landweber scheme in the pixel basis')
    parser.add_argument('--alpha_pixel', type=float, default=None, nargs='*', help='regularization parameter for sparsity in the pixel basis')  # default: for landweber_pixel as well as landweber_wavelet
    parser.add_argument('--gamma', type=float, default=0.1, nargs='*', help='regularization parameter for heuristic volume conservation')
    parser.add_argument('--lower_bound', type=float, default=0, help='lower phyiscal bound of the height field minus the offset')
    parser.add_argument('--upper_bound', type=float, default=0.3, help='upper phyiscal bound of the height field minus the offset')
    parser.add_argument('--mask_zero_eps', type=float, default=1e-3, help='mask is 0, where the absolute value of the height field minus the offset is smaller than this')
    parser.add_argument('--mask_np', type=float, default=0.05, help='Increase mask by given amount')
    parser.add_argument('--volume_eps', type=float, default=0.25, help='max relative error of printed volume')
    parser.add_argument('--volume_radius', type=int, default=2, nargs='*', help='radius in pixels for heuristic volume adaption')
    parser.add_argument('--wavelet', choices=['db3', 'coif3', 'bior1.5'], default='db3', help='wavelet family')
    parser.add_argument('--tau_wavelet', type=float, default=1e-1, nargs='*', help='step size for the Landweber scheme in the wavelet basis')
    parser.add_argument('--alpha_wavelet', type=float, default=1e-2, nargs='*', help='regularization parameter for sparsity in the wavelet basis')

    args_list = parser.parse_args()

    # set default device and dtype
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_list.gpu)
    args_list.device = th.device('cuda:0')

    if args_list.data_norm == 'l1':
        args_list.objective_func = l1
    else:
        args_list.objective_func = l2_sq

    # set unset parameters to appropriate defaults
    if args_list.alpha_pixel is None:
        args_list.alpha_pixel = 2e-3 if args_list.reconstruction_method == 'landweber_pixel' else 5e-4

    if args_list.tv_eps is None:
        args_list.tv_eps = args_list.noise_level / 2 if type(args_list.noise_level) is not list else [l / 2 for l in args_list.noise_level]

    if isinstance(args_list.additional_elements, str):
        args_list.additional_elements = [args_list.additional_elements]

    # set useful information
    args_list.commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    args_list.user = getpass.getuser()
    args_list.argument_list = str(sys.argv)

    product_parameters = ['height_offset',
                          'screen_position',
                          'num_simulations_reference',
                          'num_inner_simulations_reference',
                          'num_simulations',
                          'num_inner_simulations',
                          'height_field_resolution',
                          'photon_map_size_reference',
                          'photon_map_size',
                          'splat_smoothing_reference',
                          'splat_smoothing',
                          'max_pixel_radius',
                          'noise_level',
                          'tau_dis',
                          'tau_pixel',
                          'alpha_pixel',
                          'gamma',
                          'beta_pixel',
                          'tv_eps',
                          'alpha_wavelet',
                          'tau_wavelet',
                          'volume_radius']

    filtered_params = [p for p in product_parameters if type(args_list.__dict__[p]) is list]

    import copy
    args = copy.deepcopy(args_list)
    if len(filtered_params) == 0:
        args.datetime = get_datetime_identifier()
        yield args
    elif len(filtered_params) == 1:
        for p in args_list.__dict__[filtered_params[0]]:
            args.__dict__[filtered_params[0]] = p
            args.datetime = get_datetime_identifier()
            yield args
    else:
        for p in product(*(args_list.__dict__[param_string] for param_string in filtered_params)):
            args.__dict__.update(dict(zip(filtered_params, p)))
            args.datetime = get_datetime_identifier()
            yield args
