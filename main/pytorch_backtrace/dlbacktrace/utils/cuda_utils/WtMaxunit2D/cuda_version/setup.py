import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))

include_dirs = [osp.join(this_dir, "include")]

setup(
    name='wt_maxunit2d_ops',
    version='0.1',
    description='Weighted-Fused-MaxUnit2D operations',
    author='',
    author_email='',
    url='',
    ext_modules=[CUDAExtension(
        name='wt_maxunit2d_ops',
        sources=['calculate_wt_maxunit2d_op.cpp', 'calculate_wt_maxunit2d_kernel.cu'],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-O2'],
                            'nvcc': ['-O2', '--use_fast_math']},
    )],
    cmdclass={
        'build_ext': BuildExtension
    }
)
