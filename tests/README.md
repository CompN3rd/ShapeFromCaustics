# Tests for Shape from Caustics

## Running Tests

### Prerequisites
- PyTorch with CUDA support
- Built PhotonDifferentialSplatting extension (`python setup.py install`)
- pytest (`pip install pytest`)

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Module
```bash
pytest tests/test_photon_differentials.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_photon_differentials.py::TestPhotonDifferentialNumerical -v
```

### Run Specific Test Method
```bash
pytest tests/test_photon_differentials.py::TestPhotonDifferentialNumerical::test_forward_numerical_equivalence -v
```

### Show Print Statements
```bash
pytest tests/ -v -s
```

## Test Categories

### Numerical Equivalence Tests (`TestPhotonDifferentialNumerical`)
Validates that the optimized CUDA kernels produce numerically correct results by comparing against reference implementations:
- `test_forward_numerical_equivalence`: Compares optimized forward kernel against reference implementation matching the original unoptimized logic
- `test_backward_numerical_equivalence`: Compares optimized backward kernel against reference implementation
- `test_forward_backward_consistency`: Tests end-to-end gradient computation with PyTorch autograd

**Reference Implementation**: The tests include CPU-based reference implementations (`reference_pds_forward` and `reference_pds_backward`) that replicate the exact logic of the original unoptimized CUDA kernels. These serve as ground truth for validating the optimized kernels maintain numerical equivalence.

### Performance Tests (`TestPhotonDifferentialPerformance`)
Benchmarks the optimized kernels to ensure no performance regression:
- `test_forward_performance_benchmark`: Measures forward pass timing
- `test_backward_performance_benchmark`: Measures backward pass timing

Performance tests report:
- Average execution time in milliseconds
- Throughput in photons/second
- Assert no severe regression (< 100ms for 5000 photons)

### Input Validation Tests (`TestInputValidation`)
Verifies that invalid inputs are properly rejected:
- `test_forward_input_validation`: Tests forward pass input checks
- `test_backward_input_validation`: Tests backward pass input checks

## CUDA Availability

Tests are CUDA-aware and will be automatically skipped if:
- CUDA is not available on the system
- PhotonDifferentialSplatting extension is not built

Example skip message:
```
SKIPPED [1] tests/test_photon_differentials.py:20: CUDA not available
```

## Performance Benchmarking

The performance tests use small problem sizes for fast execution in CI:
- 5000 photons
- 512x512 output grid
- 5 timed runs with 2 warmup runs

For more comprehensive benchmarking, modify the test parameters:
```python
# In test_photon_differentials.py
num_photons = 50000  # Increase for larger benchmark
num_runs = 20        # More runs for better statistics
```

## Test Execution Time

Expected execution times (with CUDA):
- Numerical tests: ~2-5 seconds
- Performance tests: ~3-5 seconds
- Input validation: ~1-2 seconds
- **Total: ~10-15 seconds**

## Interpreting Results

### Numerical Tests
These should always pass. If they fail, it indicates:
- Incorrect optimization implementation
- Numerical instability issues
- Platform-specific floating-point differences

### Performance Tests
These establish baseline performance expectations:
- Actual speedup depends on GPU model
- Modern GPUs should see 20-40% improvement with optimizations
- Tests assert no *regression* rather than minimum speedup

### Validation Tests
These verify proper error handling:
- CPU tensors should be rejected
- Non-contiguous tensors should be rejected
- Error messages should mention "CUDA tensor" or "contiguous"

## Troubleshooting

### Tests are skipped
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure extension is built: `python -c "import PhotonDifferentialSplatting"`

### Out of memory errors
- Reduce problem sizes in tests
- Close other GPU applications
- Use a GPU with more VRAM

### Numerical differences
- Small differences (< 1e-5) are expected due to floating-point arithmetic
- Large differences indicate a bug in the optimizations

## Adding New Tests

When adding new tests:
1. Use `@pytest.mark.skipif` decorators for CUDA-dependent tests
2. Use deterministic seeds for reproducibility
3. Keep problem sizes small for fast execution
4. Include both correctness and performance checks
5. Document expected behavior in docstrings
