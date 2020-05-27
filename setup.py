import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rFS", 
    version="0.0.4",
    author="Owais Sarwar",
    author_email="osarwar@andrew.cmu.edu",
    description="A tool to build regression models using Regularized Forward Selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osarwar/rFS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    'numpy>=1.16.1', 
    'rpy2>=2.9.4'],
)