from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = [Extension("ris_helper", ["ris_helper.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]
ext_modules = [Extension("ris_helper", ["ris_helper.pyx"])]

setup(
    name='Ris Helper',
    cmdclass={'build_ext': build_ext},
    include_dirs=[np.get_include()],
    ext_modules=ext_modules
)

# setup(
#     ext_modules = cythonize("ris_helper.pyx"),
#     include_dirs=[np.get_include()]
# )
