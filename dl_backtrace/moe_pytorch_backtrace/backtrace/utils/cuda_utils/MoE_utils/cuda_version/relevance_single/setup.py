import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))

nvcc_flags = [
    '-O3', '--use_fast_math', '-Xcompiler', '-fPIC',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_90,code=sm_90',
    '-gencode', 'arch=compute_120,code=sm_120',
]

setup(
    name='relevance_single_ops',
    version='0.1',
    ext_modules=[CUDAExtension(
        name='relevance_single_ops',
        sources=[
            'relevance_single_op.cpp',
            'relevance_single_kernel.cu',
        ],
        extra_compile_args={
            'cxx':  ['-O3'],
            'nvcc': nvcc_flags,
        },
    )],
    cmdclass={'build_ext': BuildExtension},
)
