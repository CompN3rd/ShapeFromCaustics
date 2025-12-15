"""
Tests for photon differential splatting CUDA kernels.

This module contains tests to validate:
1. Numerical equivalence of optimized kernels against original unoptimized CUDA kernels
2. Performance benchmarks to ensure no regression

Tests are CUDA-aware and will skip gracefully if CUDA is not available.
"""

import pytest
import torch
import time

# Try to import the extension
try:
    import PhotonDifferentialSplatting as pds
    EXTENSION_AVAILABLE = True
except ImportError:
    EXTENSION_AVAILABLE = False
    pds = None

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available() if EXTENSION_AVAILABLE else False


@pytest.mark.skipif(not EXTENSION_AVAILABLE, reason="PhotonDifferentialSplatting extension not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPhotonDifferentialNumerical:
    """Test numerical equivalence of optimized kernels against original unoptimized CUDA kernels."""
    
    def create_test_inputs(self, num_photons=100, seed=42):
        """Create deterministic test inputs for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Create test tensors
        Ep = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 10.0
        xp = torch.rand(2, num_photons, dtype=torch.float32, device='cuda') * 2.0 - 1.0
        Mp = torch.rand(2, 3, num_photons, dtype=torch.float32, device='cuda')
        cp = torch.randint(0, 3, (1, num_photons), dtype=torch.int64, device='cuda')
        radius = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 0.5 + 0.01
        
        return Ep, xp, Mp, cp, radius
    
    def test_forward_numerical_equivalence(self):
        """Test that optimized forward pass produces numerically equivalent results to original CUDA kernel."""
        # Create test inputs
        Ep, xp, Mp, cp, radius = self.create_test_inputs(num_photons=50)
        
        # Set up output parameters
        output_size = [3, 256, 256]
        max_pixel_radius = 50
        
        # Run optimized CUDA kernel
        result_optimized = pds.pds_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)
        pds_grid_optimized = result_optimized[0]
        
        # Run original unoptimized CUDA kernel
        result_original = pds.pds_forward_original(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)
        pds_grid_original = result_original[0]
        
        # Validate output shape and properties
        assert pds_grid_optimized.shape == torch.Size(output_size), f"Output shape mismatch: {pds_grid_optimized.shape} vs {output_size}"
        assert pds_grid_optimized.is_cuda, "Output should be on CUDA"
        assert pds_grid_optimized.dtype == torch.float32, "Output dtype should be float32"
        assert torch.all(pds_grid_optimized >= 0), "Output should be non-negative"
        assert torch.isfinite(pds_grid_optimized).all(), "Output should not contain NaN or Inf"
        
        # Compare optimized vs original unoptimized CUDA kernel
        # Allow tolerance appropriate for float32 and atomic operations
        assert torch.allclose(pds_grid_optimized, pds_grid_original, rtol=1e-5, atol=1e-6), \
            f"Optimized kernel output does not match original unoptimized CUDA kernel.\n" \
            f"Max difference: {torch.max(torch.abs(pds_grid_optimized - pds_grid_original)).item()}\n" \
            f"Mean difference: {torch.mean(torch.abs(pds_grid_optimized - pds_grid_original)).item()}"
        
        # Test reproducibility of optimized kernel
        result2 = pds.pds_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)
        pds_grid2 = result2[0]
        assert torch.allclose(pds_grid_optimized, pds_grid2, rtol=1e-7, atol=1e-8), \
            "Optimized forward pass should be reproducible with same inputs"
    
    def test_backward_numerical_equivalence(self):
        """Test that optimized backward pass produces numerically equivalent results to original CUDA kernel."""
        # Create test inputs
        Ep, xp, Mp, cp, radius = self.create_test_inputs(num_photons=50)
        
        # Create gradient input
        grad_pds = torch.rand(3, 256, 256, dtype=torch.float32, device='cuda')
        max_pixel_radius = 50
        
        # Run optimized CUDA kernel
        result_optimized = pds.pds_backward(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius)
        grad_Ep_opt, grad_xp_opt, grad_Mp_opt = result_optimized
        
        # Run original unoptimized CUDA kernel
        result_original = pds.pds_backward_original(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius)
        grad_Ep_orig, grad_xp_orig, grad_Mp_orig = result_original
        
        # Validate output structure
        assert len(result_optimized) == 3, "Backward should return gradients for Ep, xp, Mp"
        
        # Validate shapes
        assert grad_Ep_opt.shape == Ep.shape, f"grad_Ep shape mismatch: {grad_Ep_opt.shape} vs {Ep.shape}"
        assert grad_xp_opt.shape == xp.shape, f"grad_xp shape mismatch: {grad_xp_opt.shape} vs {xp.shape}"
        assert grad_Mp_opt.shape == Mp.shape, f"grad_Mp shape mismatch: {grad_Mp_opt.shape} vs {Mp.shape}"
        
        # Validate properties
        assert grad_Ep_opt.is_cuda, "grad_Ep should be on CUDA"
        assert grad_xp_opt.is_cuda, "grad_xp should be on CUDA"
        assert grad_Mp_opt.is_cuda, "grad_Mp should be on CUDA"
        
        assert torch.isfinite(grad_Ep_opt).all(), "grad_Ep should not contain NaN or Inf"
        assert torch.isfinite(grad_xp_opt).all(), "grad_xp should not contain NaN or Inf"
        assert torch.isfinite(grad_Mp_opt).all(), "grad_Mp should not contain NaN or Inf"
        
        # Compare optimized vs original unoptimized CUDA kernel
        # Allow tolerance appropriate for float32
        assert torch.allclose(grad_Ep_opt, grad_Ep_orig, rtol=1e-5, atol=1e-6), \
            f"Optimized grad_Ep does not match original unoptimized CUDA kernel.\n" \
            f"Max diff: {torch.max(torch.abs(grad_Ep_opt - grad_Ep_orig)).item()}"
        
        assert torch.allclose(grad_xp_opt, grad_xp_orig, rtol=1e-5, atol=1e-6), \
            f"Optimized grad_xp does not match original unoptimized CUDA kernel.\n" \
            f"Max diff: {torch.max(torch.abs(grad_xp_opt - grad_xp_orig)).item()}"
        
        assert torch.allclose(grad_Mp_opt, grad_Mp_orig, rtol=1e-5, atol=1e-6), \
            f"Optimized grad_Mp does not match original unoptimized CUDA kernel.\n" \
            f"Max diff: {torch.max(torch.abs(grad_Mp_opt - grad_Mp_orig)).item()}"
        
        # Test reproducibility of optimized kernel
        result2 = pds.pds_backward(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius)
        grad_Ep2, grad_xp2, grad_Mp2 = result2
        
        assert torch.allclose(grad_Ep_opt, grad_Ep2, rtol=1e-7, atol=1e-8), \
            "Backward pass grad_Ep should be reproducible"
        assert torch.allclose(grad_xp_opt, grad_xp2, rtol=1e-7, atol=1e-8), \
            "Backward pass grad_xp should be reproducible"
        assert torch.allclose(grad_Mp_opt, grad_Mp2, rtol=1e-7, atol=1e-8), \
            "Backward pass grad_Mp should be reproducible"
    
    def test_forward_backward_consistency(self):
        """Test that forward and backward passes are consistent."""
        # Create test inputs with requires_grad
        num_photons = 30
        Ep_in = torch.rand(num_photons, dtype=torch.float32, device='cuda', requires_grad=True)
        xp_in = torch.rand(2, num_photons, dtype=torch.float32, device='cuda', requires_grad=True)
        Mp_in = torch.rand(2, 3, num_photons, dtype=torch.float32, device='cuda', requires_grad=True)
        cp_in = torch.randint(0, 3, (1, num_photons), dtype=torch.int64, device='cuda')
        radius_in = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 0.3 + 0.05
        
        output_size = [3, 128, 128]
        max_pixel_radius = 30
        
        # Forward pass
        result = pds.pds_forward(Ep_in, xp_in, Mp_in, cp_in, radius_in, output_size, max_pixel_radius)
        pds_grid = result[0]
        
        # Create a simple loss
        loss = pds_grid.sum()
        
        # Compute gradients using PyTorch autograd
        loss.backward()
        
        # Check that gradients were computed
        assert Ep_in.grad is not None, "Ep should have gradients"
        assert xp_in.grad is not None, "xp should have gradients"
        assert Mp_in.grad is not None, "Mp should have gradients"
        
        # Check gradients are finite
        assert torch.isfinite(Ep_in.grad).all(), "Ep gradients should be finite"
        assert torch.isfinite(xp_in.grad).all(), "xp gradients should be finite"
        assert torch.isfinite(Mp_in.grad).all(), "Mp gradients should be finite"


@pytest.mark.skipif(not EXTENSION_AVAILABLE, reason="PhotonDifferentialSplatting extension not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestPhotonDifferentialPerformance:
    """Test performance of optimized kernels."""
    
    def create_benchmark_inputs(self, num_photons=10000):
        """Create representative inputs for benchmarking."""
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        
        Ep = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 10.0
        xp = torch.rand(2, num_photons, dtype=torch.float32, device='cuda') * 2.0 - 1.0
        Mp = torch.rand(2, 3, num_photons, dtype=torch.float32, device='cuda')
        cp = torch.randint(0, 3, (1, num_photons), dtype=torch.int64, device='cuda')
        radius = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 0.3 + 0.05
        
        return Ep, xp, Mp, cp, radius
    
    def benchmark_forward(self, num_photons=10000, num_runs=10, warmup=3):
        """Benchmark forward pass performance."""
        Ep, xp, Mp, cp, radius = self.create_benchmark_inputs(num_photons)
        output_size = [3, 512, 512]
        max_pixel_radius = 100
        
        # Warmup runs
        for _ in range(warmup):
            _ = pds.pds_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)
        torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        for _ in range(num_runs):
            result = pds.pds_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def benchmark_backward(self, num_photons=10000, num_runs=10, warmup=3):
        """Benchmark backward pass performance."""
        Ep, xp, Mp, cp, radius = self.create_benchmark_inputs(num_photons)
        grad_pds = torch.rand(3, 512, 512, dtype=torch.float32, device='cuda')
        max_pixel_radius = 100
        
        # Warmup runs
        for _ in range(warmup):
            _ = pds.pds_backward(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius)
        torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        for _ in range(num_runs):
            result = pds.pds_backward(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def test_forward_performance_benchmark(self):
        """Benchmark forward pass and report timing."""
        # Use smaller problem size for quick test
        num_photons = 5000
        num_runs = 5
        
        avg_time = self.benchmark_forward(num_photons=num_photons, num_runs=num_runs, warmup=2)
        
        print(f"\n=== Forward Pass Benchmark ===")
        print(f"Photons: {num_photons}")
        print(f"Average time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {num_photons/avg_time:.0f} photons/sec")
        
        # Performance assertion: should complete in reasonable time
        # For 5000 photons, expect < 100ms per forward pass
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time*1000:.2f} ms > 100 ms"
    
    def test_backward_performance_benchmark(self):
        """Benchmark backward pass and report timing."""
        # Use smaller problem size for quick test
        num_photons = 5000
        num_runs = 5
        
        avg_time = self.benchmark_backward(num_photons=num_photons, num_runs=num_runs, warmup=2)
        
        print(f"\n=== Backward Pass Benchmark ===")
        print(f"Photons: {num_photons}")
        print(f"Average time: {avg_time*1000:.2f} ms")
        print(f"Throughput: {num_photons/avg_time:.0f} photons/sec")
        
        # Performance assertion: should complete in reasonable time
        # For 5000 photons, expect < 100ms per backward pass
        assert avg_time < 0.1, f"Backward pass too slow: {avg_time*1000:.2f} ms > 100 ms"


@pytest.mark.skipif(not EXTENSION_AVAILABLE, reason="PhotonDifferentialSplatting extension not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestInputValidation:
    """Test input validation in C++ wrappers."""
    
    def test_forward_input_validation(self):
        """Test that forward pass validates inputs properly."""
        # Create valid inputs
        num_photons = 10
        Ep = torch.rand(num_photons, dtype=torch.float32, device='cuda')
        xp = torch.rand(2, num_photons, dtype=torch.float32, device='cuda')
        Mp = torch.rand(2, 3, num_photons, dtype=torch.float32, device='cuda')
        cp = torch.randint(0, 3, (1, num_photons), dtype=torch.int64, device='cuda')
        radius = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 0.3
        
        output_size = [3, 64, 64]
        max_pixel_radius = 20
        
        # Valid call should work
        result = pds.pds_forward(Ep, xp, Mp, cp, radius, output_size, max_pixel_radius)
        assert result is not None
        
        # Test CPU tensor rejection
        Ep_cpu = Ep.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_forward(Ep_cpu, xp, Mp, cp, radius, output_size, max_pixel_radius)
        
        # Test non-contiguous tensor rejection
        xp_non_contig = xp.transpose(0, 1)
        with pytest.raises(RuntimeError, match="must be contiguous"):
            pds.pds_forward(Ep, xp_non_contig, Mp, cp, radius, output_size, max_pixel_radius)
    
    def test_backward_input_validation(self):
        """Test that backward pass validates inputs properly."""
        # Create valid inputs
        num_photons = 10
        grad_pds = torch.rand(3, 64, 64, dtype=torch.float32, device='cuda')
        Ep = torch.rand(num_photons, dtype=torch.float32, device='cuda')
        xp = torch.rand(2, num_photons, dtype=torch.float32, device='cuda')
        Mp = torch.rand(2, 3, num_photons, dtype=torch.float32, device='cuda')
        cp = torch.randint(0, 3, (1, num_photons), dtype=torch.int64, device='cuda')
        radius = torch.rand(num_photons, dtype=torch.float32, device='cuda') * 0.3
        max_pixel_radius = 20
        
        # Valid call should work
        result = pds.pds_backward(grad_pds, Ep, xp, Mp, cp, radius, max_pixel_radius)
        assert result is not None
        assert len(result) == 3
        
        # Test CPU tensor rejection for grad_pds
        grad_pds_cpu = grad_pds.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_backward(grad_pds_cpu, Ep, xp, Mp, cp, radius, max_pixel_radius)
        
        # Test CPU tensor rejection for Ep
        Ep_cpu = Ep.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_backward(grad_pds, Ep_cpu, xp, Mp, cp, radius, max_pixel_radius)
        
        # Test CPU tensor rejection for xp
        xp_cpu = xp.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_backward(grad_pds, Ep, xp_cpu, Mp, cp, radius, max_pixel_radius)
        
        # Test CPU tensor rejection for Mp
        Mp_cpu = Mp.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_backward(grad_pds, Ep, xp, Mp_cpu, cp, radius, max_pixel_radius)
        
        # Test CPU tensor rejection for cp
        cp_cpu = cp.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_backward(grad_pds, Ep, xp, Mp, cp_cpu, radius, max_pixel_radius)
        
        # Test CPU tensor rejection for radius
        radius_cpu = radius.cpu()
        with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
            pds.pds_backward(grad_pds, Ep, xp, Mp, cp, radius_cpu, max_pixel_radius)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
