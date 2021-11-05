import os
from distutils.core import setup
from subprocess import getoutput

import setuptools


def get_version_tag() -> str:
    try:
        version = os.environ["FFT_CONV_PYTORCH_VERSION"]
    except KeyError:
        version = getoutput("git describe --tags --abbrev=0")

    return version


setup(
    name="fft-conv-pytorch",
    version=get_version_tag(),
    author="Frank Odom",
    author_email="frank.odom.iii@gmail.com",
    url="https://github.com/fkodom/fft-conv-pytorch",
    packages=setuptools.find_packages(exclude=["tests"]),
    description="Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "torch>=1.7",
    ],
    extras_require={
        "test": [
            "black",
            "flake8",
            "isort",
            "pytest",
            "pytest-cov",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
