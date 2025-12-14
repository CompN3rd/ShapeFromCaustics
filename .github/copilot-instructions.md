# GitHub Copilot Instructions for ShapeFromCaustics

## Project Overview

This is a PyTorch implementation for **Shape from Caustics**, a method for reconstructing 3D-printed glass objects from simulated caustic images using differentiable ray tracing. The project combines:

- **PyTorch** for deep learning and optimization
- **NVIDIA OptiX 6.5** for GPU-accelerated ray tracing
- **CUDA extensions** for custom differentiable rendering kernels
- **C++** for performance-critical components

Reference: [Shape from Caustics: Reconstruction of 3D-Printed Glass from Simulated Caustic Images](https://graphics.tu-bs.de/publications/kassubeck2020shape) (WACV 2021)

## Architecture

### Core Components

1. **PyOptix/** - C++/CUDA extensions for OptiX ray tracing
   - `RayTrace.cpp` - Main OptiX wrapper
   - `PhotonDifferentialSplattig.cpp` - Photon differential splatting (note: filename has typo "Splattig")
   - `kernel/photon_differentials.cu` - CUDA kernels for differentiable rendering
   - `kernel/ray_programs.cu` - OptiX ray programs

2. **model/** - Python modules for caustics simulation
   - `caustics.py` - Caustics computation and refraction
   - `photon_differential.py` - Photon differential computations
   - `renderable_object.py` - 3D object representations
   - `utils.py` - Utilities including TensorBoard logging

3. **shape_from_caustics.py** - Main entry point for simulation and reconstruction
4. **hyperparameter_helper.py** - Command-line argument parsing
5. **schwartzburg_2014/** - Reimplementation of prior work

## Technology Stack

- **Python 3.7-3.8** (see README for full version support)
- **PyTorch >= 1.7** with CUDA support
- **NVIDIA OptiX 6.5** (legacy version)
- **CUDA** for GPU acceleration
- **CMake** for building OptiX PTX files
- **PyTorch C++/CUDA extensions** via setuptools

### Key Python Dependencies
- PyWavefront (mesh loading)
- pytorch_wavelets (wavelet sparsity)
- PyWavelets
- Matplotlib (visualization)
- tqdm (progress bars)
- TensorBoard (logging)
- PyMongeAmpere (for Schwartzburg 2014 method)

## Build System

### Building OptiX PTX Files
```bash
mkdir build && cd build
cmake ..
cmake --build . --target install
```
The CMake build generates `ray_programs.ptx` required for OptiX ray tracing.

### Building Python Extensions
```bash
python setup.py install
```
**Important**: Update OptiX paths in `setup.py` before building:
- Windows: `<path_to_optix>/OptiX SDK 6.5.0/`
- Linux: `<path_to_optix>/NVIDIA-OptiX-SDK-6.5.0-linux64/`

## Coding Conventions

### C++ Style
- Follow **Google C++ Style Guide** (with modifications)
- Use `.clang-format` configuration in repository root
- Key settings:
  - Indent width: 4 spaces (tabs for indentation)
  - Column limit: 200
  - Pointer alignment: left (`Type* ptr`)
  - Brace wrapping: custom (braces on new line for classes, functions, namespaces)
  - Constructor initializers: after colon

### Python Style
- Follow **PEP 8** conventions
- Use PyTorch tensor operations (`th` alias for `torch`)
- Prefer functional programming style where applicable
- Use descriptive variable names

### CUDA Conventions
- Use `constexpr` for compile-time constants
- Default block size: 256 threads (see `kernel/photon_differentials.cu`)
- Use `--use_fast_math` for CUDA compilation

## File Organization

- **Build artifacts**: `build/`, `dist/`, `*.egg-info/` (in .gitignore)
- **Compiled files**: `*.so`, `*.ptx`, `*.o`, `*.obj` (in .gitignore)
- **Output files**: `runs/` (plots), `savestates/` (checkpoints)
- **Input data**: `img/` (images)
- **Temporary files**: Use `/tmp` for temporary files during development

## Running the Code

### Basic Usage
```bash
python shape_from_caustics.py --help  # See all parameters
python shape_from_caustics.py         # Run with default parameters
```

### Key Parameters
- `--height_field_resolution` - Resolution of the height field
- `--num_inner_simulations` - Number of inner simulations (reduce if OOM)
- See `hyperparameter_helper.py` for complete parameter list

### Memory Considerations
- Default parameters assume 24GB VRAM
- Reduce `num_inner_simulations` if encountering OOM errors

## Testing

**Note**: This repository does not currently have a formal test suite. When adding tests:
- Follow PyTorch testing conventions
- Use `pytest` if adding test infrastructure
- Consider CUDA availability for GPU-dependent tests

## Dependencies and Security

- OptiX SDK requires manual installation and path configuration
- Ensure OptiX shared libraries are in `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux)
- No automatic package manager for OptiX - manual setup required

## Common Tasks

### Adding New Features
1. For Python modules: add to `model/` directory
2. For CUDA kernels: add to `PyOptix/kernel/`
3. Update `setup.py` if adding new C++ extensions

### Modifying Ray Tracing
1. Edit `kernel/ray_programs.cu` for OptiX programs
2. Rebuild with CMake to regenerate PTX files
3. Reinstall Python extension with `setup.py`

### Visualization Changes
- TensorBoard logging: use `model.utils.tensorboard_logger`
- Plot generation: see `post_optimization_plot()` in `shape_from_caustics.py`

## Known Constraints

- **OptiX version locked to 6.5** (legacy version, not latest)
- **Python version**: 3.7-3.8 (not tested on 3.9+)
- **Platform**: Windows and Linux supported
- **GPU**: NVIDIA GPU with CUDA support required
- **Build system**: Requires both CMake and Python setuptools

## Additional Context

- This is a research codebase from WACV 2021 publication
- Focus on differentiable rendering and inverse problems
- High computational requirements (GPU with substantial VRAM needed)
- Authors: Marc Kassubeck, Florian Bürgel
- License: MIT
