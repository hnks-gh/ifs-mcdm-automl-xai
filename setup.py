"""
setup.py — IFS-MCDM-AutoML-XAI Framework
Install in editable mode with: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="ifs_mcdm_automl_xai",
    version="0.1.0",
    description=(
        "Integrated IFS-MCDM and AutoML-XAI framework "
        "for Vietnam PAPI 2011-2024 empirical evaluation"
    ),
    author="Research Team",
    python_requires=">=3.11,<3.12",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[],          # managed via requirements.txt
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-cov>=5.0.0",
            "flake8>=7.0.0",
            "mypy>=1.10.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
