import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))

include_dirs = [osp.join(this_dir, "include")]

# Multi-architecture support: compile for common GPU architectures
nvcc_flags = [
    '-O3', '--use_fast_math', '-Xcompiler', '-fPIC',
    # Volta
    '-gencode', 'arch=compute_70,code=sm_70',
    # Turing
    '-gencode', 'arch=compute_75,code=sm_75',
    # Ampere
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    # Ada Lovelace
    '-gencode', 'arch=compute_89,code=sm_89',
    # Hopper
    '-gencode', 'arch=compute_90,code=sm_90',
]

setup(
    name='wt_fc_v3_ops',
    version='0.1',
    description='Weighted-Fused-Linear v3 operations (precompiled)',
    ext_modules=[CUDAExtension(
        name='wt_fc_v3_ops',
        sources=[
            'calculate_wt_fc_v3_op.cpp',
            'calculate_wt_fc_v3_kernel.cu'
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': nvcc_flags,
        },
    )],
    cmdclass={
        'build_ext': BuildExtension
    }
)
