#include "utils.hpp"
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;
namespace th = torch;

// CUDA forward declarations
std::vector<th::Tensor> pds_cuda_forward(th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, const std::vector<int64_t>& output_size, int32_t max_pixel_radius);
std::vector<th::Tensor> pds_cuda_backward(th::Tensor grad_pds, th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, int32_t max_pixel_radius);

// Original unoptimized CUDA kernels (for testing)
std::vector<th::Tensor> pds_cuda_forward_original(th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, const std::vector<int64_t>& output_size, int32_t max_pixel_radius);
std::vector<th::Tensor> pds_cuda_backward_original(th::Tensor grad_pds, th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, int32_t max_pixel_radius);

std::vector<th::Tensor> pds_forward(th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, const std::vector<int64_t>& output_size, int32_t max_pixel_radius)
{
	CHECK_INPUT(Ep);
	CHECK_INPUT(xp);
	CHECK_INPUT(Mp);
	CHECK_INPUT(cp);
	CHECK_INPUT(radius);
	TORCH_CHECK(Ep.size(0) == xp.size(1) && Ep.size(0) == Mp.size(2) && Ep.size(0) == cp.size(1) && Ep.size(0) == radius.size(0), "Dimensions of tensors for pds_forward don't match");
	TORCH_CHECK(cp.size(0) == 1, "Channel index must be one dimensional!");
	TORCH_CHECK(xp.size(0) == 2, "Position index must be two dimensional!");
	TORCH_CHECK(Mp.size(0) == 2 && Mp.size(1) == 3, "Change of basis matrix must be 2x3!");

	return pds_cuda_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius);
}

std::vector<th::Tensor> pds_backward(th::Tensor grad_pds, th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, int32_t max_pixel_radius)
{
	CHECK_INPUT(grad_pds);
	CHECK_INPUT(Ep);
	CHECK_INPUT(xp);
	CHECK_INPUT(Mp);
	CHECK_INPUT(cp);
	CHECK_INPUT(radius);

	return pds_cuda_backward(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius);
}

std::vector<th::Tensor> pds_forward_original(th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, const std::vector<int64_t>& output_size, int32_t max_pixel_radius)
{
	CHECK_INPUT(Ep);
	CHECK_INPUT(xp);
	CHECK_INPUT(Mp);
	CHECK_INPUT(cp);
	CHECK_INPUT(radius);
	TORCH_CHECK(Ep.size(0) == xp.size(1) && Ep.size(0) == Mp.size(2) && Ep.size(0) == cp.size(1) && Ep.size(0) == radius.size(0), "Dimensions of tensors for pds_forward don't match");
	TORCH_CHECK(cp.size(0) == 1, "Channel index must be one dimensional!");
	TORCH_CHECK(xp.size(0) == 2, "Position index must be two dimensional!");
	TORCH_CHECK(Mp.size(0) == 2 && Mp.size(1) == 3, "Change of basis matrix must be 2x3!");

	return pds_cuda_forward_original(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius);
}

std::vector<th::Tensor> pds_backward_original(th::Tensor grad_pds, th::Tensor Ep, th::Tensor xp, th::Tensor Mp, th::Tensor cp, th::Tensor radius, int32_t max_pixel_radius)
{
	CHECK_INPUT(grad_pds);
	CHECK_INPUT(Ep);
	CHECK_INPUT(xp);
	CHECK_INPUT(Mp);
	CHECK_INPUT(cp);
	CHECK_INPUT(radius);

	return pds_cuda_backward_original(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("pds_forward", &pds_forward, "PDS forward (CUDA)");
	m.def("pds_backward", &pds_backward, "PDS backward (CUDA)");
	m.def("pds_forward_original", &pds_forward_original, "PDS forward original unoptimized (CUDA)");
	m.def("pds_backward_original", &pds_backward_original, "PDS backward original unoptimized (CUDA)");
}