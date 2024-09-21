from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

package_dir = os.path.join("src", "preprocess_cython")

extensions = [
    Extension(
        "preprocess_cython.preprocess",              
        [os.path.join(package_dir, "preprocess.pyx")], 
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions),
    package_dir={"": "src"},  
    packages=["preprocess_cython"],  
)
