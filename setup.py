from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
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
    packages=find_packages(where="src"),
    include_package_data=True,
)
