""" setuptools-based setup module. """

from setuptools import setup, find_packages

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

setup(
    name="tomcat_speech",
    version="0.1",
    description="Speech analysis tools for the ToMCAT project",
    url="https://github.com/clulab/tomcat-speech",
    packages = find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="speech analysis",
    zip_safe=False,
    install_requires=[
        "wheel",
        "torch",
        "torchtext",
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "tqdm"
    ],
    python_requires=">=3.7",
)

