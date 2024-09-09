from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="preprocess",
        sources=["preprocess.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="Preprocess",
    ext_modules=cythonize(extensions),
)
