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
<<<<<<< HEAD
        "torch", #without version specification, it was set to 1.9.0
        "torchtext", #without version specification, it was set to 0.10
=======
        "torch", #it was set to torch==1.9.0
        "torchtext==0.10",
>>>>>>> e0b0cfe7eb82d5b06c9a55856d0028ef5cbc9d56
        "torchaudio",
        "nltk",
        "librosa",
        "h5py",
        "pandas",
        "numpy",
<<<<<<< HEAD
        "scikit-learn", #changed to scikit-learn
=======
        "sklearn-learn",
>>>>>>> e0b0cfe7eb82d5b06c9a55856d0028ef5cbc9d56
        "matplotlib",
        "tqdm",
        "webvtt-py",
        "transformers"
    ],
    extras_require={"mmc_server": ["uvicorn", "fastapi"]},
    python_requires=">=3.7",
)
