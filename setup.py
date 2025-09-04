"""
Setup script for GACL Wildlife Classification package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gacl-wildlife-classification",
    version="1.0.0",
    author="SPHERE AX AILab",
    author_email="yuncs@sphereax.com",
    description="Graph Attention Contrastive Learning for Korean Wildlife Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/changsubi/GACL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.982",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "tensorboard": [
            "tensorboard>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gacl-train=scripts.train:main",
            "gacl-inference=scripts.inference:main",
        ],
    },
)
