import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))

include_dirs = [osp.join(this_dir, "include")]

setup(
    name='wt_fc_ops',
    version='0.1',
    description='Weighted-Fused-Linear operations',
    author='',
    author_email='',
    url='',
    ext_modules=[CUDAExtension(
        name='wt_fc_ops',
        sources=[
                'calculate_wt_fc_op.cpp',
                'calculate_wt_fc_kernel.cu'
                ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3', '--use_fast_math', '-Xcompiler', '-fPIC']},
    )],
    cmdclass={
        'build_ext': BuildExtension
    }
)
