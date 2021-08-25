""" setuptools-based setup module. """

from setuptools import setup, find_packages

from setuptools import setup

setup(
    name="tomcat_speech",
    version="0.2",
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
        "tqdm",
        "webvtt-py"
    ],
    extras_require={"mmc_server": ["uvicorn", "fastapi"]},
    python_requires=">=3.7",
)
