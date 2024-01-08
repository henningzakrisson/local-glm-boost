from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="local-glm-boost",
    version="0.0.1",
    author="Henning Zakrisson",
    author_email="henning.zakrisson@gmail.com",
    description="A python package for the LocalGLMboost algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/henningzakrisson/local-glm-boost",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    license="MIT",
    license_files=("LICENSE",),
)
