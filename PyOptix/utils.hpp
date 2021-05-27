#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// some sanity defines
// --------------------------------------------------------------------------------------

#define CUDACHECKERROR(err)                                                                           \
	do {                                                                                              \
		if (err != cudaSuccess) {                                                                     \
			std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
			std::terminate();                                                                         \
		}                                                                                             \
	} while (false);

inline void checkCudaErrorsHelper(const char* file, int line, bool abort = true)
{
	cudaError_t result = cudaSuccess;
	// wait only if we have time to do so (i.e. if we are debugging)
#ifdef NDEBUG
	result = cudaGetLastError();
#else
	result = cudaDeviceSynchronize();
#endif  // !NDEBUG

	if (result != cudaSuccess) {
		std::cerr << "CUDA Launch Error: " << cudaGetErrorString(result) << " in " << file << " at " << line << std::endl;
		if (abort) std::terminate();
	}
}

#define CHECKCUDAERRORS()                          \
	{                                              \
		checkCudaErrorsHelper(__FILE__, __LINE__); \
	}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
	CHECK_CUDA(x);     \
	CHECK_CONTIGUOUS(x)
// --------------------------------------------------------------------------------------