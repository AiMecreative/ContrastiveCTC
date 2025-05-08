from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="ctc_mask_cuda",
    ext_modules=[
        CUDAExtension(
            name="ctc_mask_cuda",
            sources=["ctc_cu/ctc_mask.cpp", "ctc_cu/ctc_mask_cu.cu"],
            extra_compile_args=["-std=c++17"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
