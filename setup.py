from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import subprocess
import os

# build and install optixPTXPrograms.ptx
subprocess.run(["cmake", "--build", "./build", "--target", "install", "--config", "Release"])

setup(name="PyOptix",
      ext_modules=[CUDAExtension(
          name="PyOptix",
          sources=["PyOptix/RayTrace.cpp"],
          # CHANGE PATH
          library_dirs=['<path_to_optix>/OptiX SDK 6.5.0/lib64'] if os.name == 'nt' else ['<path_to_optix>/NVIDIA-OptiX-SDK-6.5.0-linux64/lib64'],
          extra_objects=["optix.6.5.0.lib", "optixu.6.5.0.lib"] if os.name == 'nt' else [],
          libraries=[] if os.name == 'nt' else ["optix", "optixu"],
          define_macros=[("NOMINMAX", "1")] if os.name == 'nt' else []
      ),
          CUDAExtension(
              name="PhotonDifferentialSplatting",
              sources=["PyOptix/PhotonDifferentialSplattig.cpp", "PyOptix/kernel/photon_differentials.cu"],
      )],
      cmdclass={'build_ext': BuildExtension},
      data_files=[("ptx_files", ["PyOptix/ray_programs.ptx"])],
      # CHANGE PATH
      include_dirs=['<path_to_optix>/OptiX SDK 6.5.0/include'] if os.name == 'nt' else ['<path_to_optix>/NVIDIA-OptiX-SDK-6.5.0-linux64/include'],
      version='1.0.0',
      author="Marc Kassubeck",
      author_email="kassubeck@cg.cs.tu-bs.de"
      )
