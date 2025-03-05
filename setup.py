from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Circuit_Regressor",
    version="0.1.0",
    author="Carlos Murillo",
    author_email="cmurillos@unal.edu.co",
    description="Given an AC impedance-frequency curve, equivalent circuits are calculated",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cmurillos/Circuit_Regressor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "tensorflow>=2.8.0",
        "sympy>=1.8.0",
        "shapely>=1.8.0",
        "tqdm>=4.62.0",
    ],
)

