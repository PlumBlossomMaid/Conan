"""Setup script for Conan."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion"

setup(
    name="conan-vc",
    version="0.1.0",
    author="Yu Zhang, Baotong Tian, Zhiyao Duan",
    author_email="",
    description="A chunkwise online network for zero-shot adaptive voice conversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PlumBlossomMaid/Conan",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "paddleocean @ git+https://github.com/PlumBlossomMaid/PaddleOcean.git",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "numpy>=1.24.0,<2.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "onnx": ["onnx>=1.14.0"],
        "dev": ["black>=23.0.0", "flake8>=6.0.0", "pytest>=7.3.0"],
    },
    entry_points={
        "console_scripts": [
            "conan-preprocess=entry.preprocess:main",
            "conan-train=entry.train:main",
            "conan-infer=entry.infer_conan:main",
        ],
    },
)
