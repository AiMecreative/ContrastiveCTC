cmake_minimum_required(VERSION 3.26)
project(ContrastiveCTC)

if(NOT DEFINED CONDA_ENV_PATH)
    message(FATAL_ERROR "CONDA_ENV_PATH not defined. Please configure via CMakePresets.json.")
endif()
message("conda env path: ${CONDA_ENV_PATH}")

set(CMAKE_PREFIX_PATH "${CONDA_ENV_PATH}/lib/python3.10/site-packages/torch/share/cmake/Torch")
message("torch path: ${CMAKE_PREFIX_PATH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# 添加 runtime search 路径
set(CMAKE_INSTALL_RPATH
  "${CONDA_ENV_PATH}/lib"
  "${CONDA_ENV_PATH}/lib/python3.10/site-packages/torch/lib"
  "/usr/local/cuda-12.2/lib64"
)
include_directories(${CONDA_ENV_PATH}/include/python3.10)

find_package(Torch REQUIRED)

add_executable(main minimum.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}")
target_compile_definitions(main PRIVATE "${TORCH_DEFINITIONS}")
