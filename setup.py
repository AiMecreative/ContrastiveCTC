from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="ctc_decode",
    ext_modules=[
        CUDAExtension(
            name="ctc_decode",
            sources=["ctc_decode.cpp", "ctc_decode_cuda.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)