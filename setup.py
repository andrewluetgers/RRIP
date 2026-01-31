from setuptools import setup, find_packages

setup(
    name="rrip",
    version="0.1.0",
    description="Residual Reconstruction from Interpolated Priors - Optimized whole slide tile compression",
    author="Andrew Luetgers",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "scipy>=1.7.0",
        "flask>=2.0.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "rrip=rrip.cli:main",
        ],
    },
    python_requires=">=3.7",
)
