from setuptools import setup, find_packages

setup(
    name="circuit_regressor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sympy",
        "scipy",
        "tensorflow",
        "tqdm",
    ],
    author="Carlos Murillo",
    author_email="cmurillos@unal.edu.co",
    description="A symbolic regression tool for electrical circuits",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cmurillos/Circuit_Regressor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

