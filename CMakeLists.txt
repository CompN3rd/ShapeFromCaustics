cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
project(cuda_raytracer LANGUAGES CXX CUDA)

# For find OptiX.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")

# FindOptiX.cmake sets imported targets
find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})

add_library(optixIntersect OBJECT PyOptix/kernel/ray_programs.cu) 
set_property(TARGET optixIntersect PROPERTY CUDA_PTX_COMPILATION ON)
# target_link_libraries(optixIntersect optix optixu)
target_compile_options(optixIntersect PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# install the binaries
install(FILES $<TARGET_OBJECTS:optixIntersect> DESTINATION "${CMAKE_SOURCE_DIR}/PyOptix")