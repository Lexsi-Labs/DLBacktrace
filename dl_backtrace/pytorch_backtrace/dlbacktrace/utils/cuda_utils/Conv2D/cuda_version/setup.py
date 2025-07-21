import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))

include_dirs = [osp.join(this_dir, "include")]

setup(
    name='wt_conv_ops',
    version='0.1',
    description='Weighted-Convolutional operations',
    author='',
    author_email='',
    url='',
    ext_modules=[
        CUDAExtension(
            name='wt_conv_ops',
            sources=[
                'calculate_wt_conv2d.cpp',
                'conv2d_kernel.cu'
                ],
        include_dirs=include_dirs,
        extra_compile_args={'cxx': ['-O2'],
                            'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
