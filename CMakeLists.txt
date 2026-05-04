cmake_minimum_required(VERSION 3.18)
# set(CMAKE_CUDA_ARCHITECTURES 89) # arch code for NVIDIA L40S
set(CMAKE_CUDA_ARCHITECTURES native) # used when running on cluster to handle whatever GPU is found

project(CudaMatrixProject LANGUAGES CXX CUDA)

# to make clangd LSP work
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) #
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# 2. fast_matrix_market requires C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
    GIT_TAG        v1.7.6 
)
FetchContent_MakeAvailable(fast_matrix_market)

add_executable(spmv spmv.cu kernels.cu matrix_loader.cpp)
find_package(CUDAToolkit REQUIRED)

target_link_libraries(spmv
    PRIVATE 
    m                                      
    fast_matrix_market::fast_matrix_market 
    CUDA::cusparse
)
