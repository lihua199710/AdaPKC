#!/bin/bash
# adapted from https://github.com/An01168/DCNVSS/tree/master/correlation
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu',
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args},
        include_dirs=['/home/liteng/miniconda3/envs/pkc/include/python3.7m'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
