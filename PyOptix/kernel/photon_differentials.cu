#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#include <vector>

namespace th = torch;

dim3 cuda_gridsize(int n, int threads)
{
	int k = (n - 1) / threads + 1;
	int x = k;
	int y = 1;
	if (x > 65535) {
		x = ceil(sqrt(k));
		y = (n - 1) / (x * threads) + 1;
	}
	dim3 d(x, y, 1);
	return d;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t silverman(scalar_t x_sq)
{
	if (x_sq < 1) { return scalar_t(3) / scalar_t(M_PI) * (1 - x_sq) * (1 - x_sq); }

	return 0;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t d_silverman(scalar_t x, scalar_t x_sq)
{
	if (x < 1) { return -scalar_t(12) / scalar_t(M_PI) * x * (1 - x_sq); }

	return 0;
}

template<typename scalar_t>
__device__ __forceinline__ void pixel_to_coord(const int32_t& px, const int32_t& py, const scalar_t& w, const scalar_t& h, scalar_t& cx_out, scalar_t& cy_out)
{
	cx_out = 2 * scalar_t(px) / w - 1;
	cy_out = 2 * scalar_t(py) / h - 1;
}

template<typename scalar_t>
__device__ __forceinline__ void coord_to_pixel(const scalar_t& cx, const scalar_t& cy, const scalar_t& w, const scalar_t& h, int32_t& px_out, int32_t& py_out)
{
	px_out = int32_t((0.5 * cx + 0.5) * w);
	py_out = int32_t((0.5 * cy + 0.5) * h);
}

template<typename scalar_t>
__device__ __forceinline__ void matrix_multiply(const scalar_t (&M)[6], const scalar_t& p0, const scalar_t& p1, const scalar_t& p2, scalar_t& cx_out, scalar_t& cy_out)
{
	cx_out = M[0] * p0 + M[1] * p1 + M[2] * p2;
	cy_out = M[3] * p0 + M[4] * p1 + M[5] * p2;
}

template<typename scalar_t>
__global__ void pds_cuda_forward_kernel(const th::PackedTensorAccessor32<scalar_t, 1, th::RestrictPtrTraits> Ep,
                                        const th::PackedTensorAccessor32<scalar_t, 2, th::RestrictPtrTraits> xp,
                                        const th::PackedTensorAccessor32<scalar_t, 3, th::RestrictPtrTraits> Mp,
                                        const th::PackedTensorAccessor32<int64_t, 2, th::RestrictPtrTraits>  cp,
                                        const th::PackedTensorAccessor32<scalar_t, 1, th::RestrictPtrTraits> radius,
                                        th::PackedTensorAccessor32<scalar_t, 3, th::RestrictPtrTraits>       pds_grid,
                                        int32_t                                                              max_pixel_radius)
{
	// const int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 2D grid for sizes larger than allowed
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

	if (index < Ep.size(0)) {
		int64_t channel = cp[0][index];
		if (channel < pds_grid.size(0)) {
			scalar_t w = pds_grid.size(2);
			scalar_t h = pds_grid.size(1);

			// constrain calculation by cutoff radius
			const int32_t rx = min(int32_t(ceil(radius[index] * 0.5 * w)), max_pixel_radius);
			const int32_t ry = min(int32_t(ceil(radius[index] * 0.5 * h)), max_pixel_radius);

			const scalar_t cx_center = xp[0][index];
			const scalar_t cy_center = xp[1][index];

			int32_t px_center, py_center;
			coord_to_pixel(cx_center, cy_center, w, h, px_center, py_center);
			const scalar_t E_center    = Ep[index];
			const scalar_t M_center[6] = {Mp[0][0][index], Mp[0][1][index], Mp[0][2][index], Mp[1][0][index], Mp[1][1][index], Mp[1][2][index]};

			for (int32_t y_off = -ry; y_off <= ry; y_off++) {
				for (int32_t x_off = -rx; x_off <= rx; x_off++) {
					if (scalar_t(x_off * x_off) / (0.25 * w * w) + scalar_t(y_off * y_off) / (0.25 * h * h) <= 1) {
						int32_t px = px_center + x_off;
						int32_t py = py_center + y_off;
						if (px >= 0 && py >= 0 && px < pds_grid.size(2) && py < pds_grid.size(1)) {
							scalar_t cx_diff, cy_diff;
							pixel_to_coord(px, py, w, h, cx_diff, cy_diff);
							cx_diff -= cx_center;
							cy_diff -= cy_center;

							scalar_t cx_diff_circ, cy_diff_circ;
							matrix_multiply(M_center, cx_diff, cy_diff, scalar_t(0), cx_diff_circ, cy_diff_circ);

							const scalar_t value = silverman(cx_diff_circ * cx_diff_circ + cy_diff_circ * cy_diff_circ) * E_center;
							if (value > 0) atomicAdd(&pds_grid[channel][py][px], value);
						}
					}
				}
			}
		}
	}
}

std::vector<th::Tensor> pds_cuda_forward(th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, const std::vector<int64_t>& output_size, int32_t max_pixel_radius)
{
	// create memory of appropriate output_size
	auto pds_grid = th::zeros(output_size, Ep.options());

	const int threads = 512;
	// const int blocks  = (Ep.size(0) + threads - 1) / threads;
	const dim3 blocks = cuda_gridsize(Ep.size(0), threads);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(Ep.scalar_type(), "pds_forward_cuda", ([&] {
		                                    pds_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(Ep.packed_accessor32<scalar_t, 1, th::RestrictPtrTraits>(),
		                                                                                           xp.packed_accessor32<scalar_t, 2, th::RestrictPtrTraits>(),
		                                                                                           Mp.packed_accessor32<scalar_t, 3, th::RestrictPtrTraits>(),
		                                                                                           cp.packed_accessor32<int64_t, 2, th::RestrictPtrTraits>(),
		                                                                                           radius.packed_accessor32<scalar_t, 1, th::RestrictPtrTraits>(),
		                                                                                           pds_grid.packed_accessor32<scalar_t, 3, th::RestrictPtrTraits>(),
		                                                                                           max_pixel_radius);
	                                    }));
	return {pds_grid};
}

template<typename scalar_t>
__global__ void pds_cuda_backward_kernel(const th::PackedTensorAccessor32<scalar_t, 3, th::RestrictPtrTraits> grad_pds,
                                         const th::PackedTensorAccessor32<scalar_t, 1, th::RestrictPtrTraits> Ep,
                                         const th::PackedTensorAccessor32<scalar_t, 2, th::RestrictPtrTraits> xp,
                                         const th::PackedTensorAccessor32<scalar_t, 3, th::RestrictPtrTraits> Mp,
                                         const th::PackedTensorAccessor32<int64_t, 2, th::RestrictPtrTraits>  cp,
                                         const th::PackedTensorAccessor32<scalar_t, 1, th::RestrictPtrTraits> radius,
                                         th::PackedTensorAccessor32<scalar_t, 1, th::RestrictPtrTraits>       grad_Ep,
                                         th::PackedTensorAccessor32<scalar_t, 2, th::RestrictPtrTraits>       grad_xp,
                                         th::PackedTensorAccessor32<scalar_t, 3, th::RestrictPtrTraits>       grad_Mp,
                                         int32_t                                                              max_pixel_radius)
{
	// const int index = blockIdx.x * blockDim.x + threadIdx.x;
	// 2D grid for sizes larger than allowed
	const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

	if (index < Ep.size(0)) {
		int64_t channel = cp[0][index];
		if (channel < grad_pds.size(0)) {
			scalar_t w = grad_pds.size(2);
			scalar_t h = grad_pds.size(1);

			const int32_t rx = min(int32_t(ceil(radius[index] * 0.5 * w)), max_pixel_radius);
			const int32_t ry = min(int32_t(ceil(radius[index] * 0.5 * h)), max_pixel_radius);

			const scalar_t cx_center = xp[0][index];
			const scalar_t cy_center = xp[1][index];

			int32_t px_center, py_center;
			coord_to_pixel(cx_center, cy_center, w, h, px_center, py_center);
			const scalar_t E_center    = Ep[index];
			const scalar_t M_center[6] = {Mp[0][0][index], Mp[0][1][index], Mp[0][2][index], Mp[1][0][index], Mp[1][1][index], Mp[1][2][index]};

			scalar_t g_Ep    = 0;
			scalar_t g_xp[2] = {0};
			scalar_t g_Mp[6] = {0};
			for (int32_t y_off = -ry; y_off <= ry; y_off++) {
				for (int32_t x_off = -rx; x_off <= rx; x_off++) {
					if (scalar_t(x_off * x_off) / (0.25 * w * w) + scalar_t(y_off * y_off) / (0.25 * h * h) <= 1) {
						const int32_t px = px_center + x_off;
						const int32_t py = py_center + y_off;
						if (px >= 0 && py >= 0 && px < grad_pds.size(2) && py < grad_pds.size(1)) {
							scalar_t cx_diff, cy_diff;
							pixel_to_coord(px, py, w, h, cx_diff, cy_diff);
							cx_diff -= cx_center;
							cy_diff -= cy_center;

							scalar_t cx_diff_circ, cy_diff_circ;
							matrix_multiply(M_center, cx_diff, cy_diff, scalar_t(0), cx_diff_circ, cy_diff_circ);

							const scalar_t l2_sq   = cx_diff_circ * cx_diff_circ + cy_diff_circ * cy_diff_circ;
							const scalar_t l2_norm = sqrt(l2_sq);

							const scalar_t cx_diff_circ_normed = cx_diff_circ / l2_norm;
							const scalar_t cy_diff_circ_normed = cy_diff_circ / l2_norm;

							const scalar_t g_pds         = grad_pds[channel][py][px];
							const scalar_t d_kernel_grad = d_silverman(l2_norm, l2_sq) * E_center * g_pds;

							g_Ep += silverman(l2_sq) * g_pds;
							g_xp[0] += -d_kernel_grad * (M_center[0] * cx_diff_circ_normed + M_center[3] * cy_diff_circ_normed);
							g_xp[1] += -d_kernel_grad * (M_center[1] * cx_diff_circ_normed + M_center[4] * cy_diff_circ_normed);
							// last line of matrix not relevant, as c_diff_trans_normed is 0

							g_Mp[0] += d_kernel_grad * cx_diff_circ_normed * cx_diff;
							g_Mp[1] += d_kernel_grad * cx_diff_circ_normed * cy_diff;
							// g_Mp[2] += d_kernel * cx_diff_circ_normed * 0;
							g_Mp[3] += d_kernel_grad * cy_diff_circ_normed * cx_diff;
							g_Mp[4] += d_kernel_grad * cy_diff_circ_normed * cy_diff;
							// g_Mp[5] += d_kernel * cy_diff_circ_normed * 0;
						}
					}
				}
			}
			grad_Ep[index]       = g_Ep;
			grad_xp[0][index]    = g_xp[0];
			grad_xp[1][index]    = g_xp[1];
			grad_Mp[0][0][index] = g_Mp[0];
			grad_Mp[0][1][index] = g_Mp[1];
			grad_Mp[0][2][index] = g_Mp[2];
			grad_Mp[1][0][index] = g_Mp[3];
			grad_Mp[1][1][index] = g_Mp[4];
			grad_Mp[1][2][index] = g_Mp[5];
		}
	}
}

std::vector<th::Tensor> pds_cuda_backward(th::Tensor grad_pds, th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, int32_t max_pixel_radius)
{
	auto grad_Ep = th::empty_like(Ep);
	auto grad_xp = th::empty_like(xp);
	auto grad_Mp = th::empty_like(Mp);

	const int threads = 512;
	// const int blocks  = (Ep.size(0) + threads - 1) / threads;
	const dim3 blocks = cuda_gridsize(Ep.size(0), threads);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(Ep.scalar_type(), "pds_backward_cuda", ([&] {
		                                    pds_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(grad_pds.packed_accessor32<scalar_t, 3, th::RestrictPtrTraits>(),
		                                                                                            Ep.packed_accessor32<scalar_t, 1, th::RestrictPtrTraits>(),
		                                                                                            xp.packed_accessor32<scalar_t, 2, th::RestrictPtrTraits>(),
		                                                                                            Mp.packed_accessor32<scalar_t, 3, th::RestrictPtrTraits>(),
		                                                                                            cp.packed_accessor32<int64_t, 2, th::RestrictPtrTraits>(),
		                                                                                            radius.packed_accessor32<scalar_t, 1, th::RestrictPtrTraits>(),
		                                                                                            grad_Ep.packed_accessor32<scalar_t, 1, th::RestrictPtrTraits>(),
		                                                                                            grad_xp.packed_accessor32<scalar_t, 2, th::RestrictPtrTraits>(),
		                                                                                            grad_Mp.packed_accessor32<scalar_t, 3, th::RestrictPtrTraits>(),
		                                                                                            max_pixel_radius);
	                                    }));

	return {grad_Ep, grad_xp, grad_Mp};
}