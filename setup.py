"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="keras-surgeon",  # Required
    version="0.2.0",  # Required
    description="A library for performing network surgery on trained Keras models. "
    "Useful for deep neural network pruning.",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.10, <4",
    # only functional dependencies, no dev dependencies (like pytest)
    install_requires=["importlib-metadata", "pandas", "pillow", "tensorflow"],
    extras_require={  # Optional
        "dev": ["build"],
        "test": ["pytest"],
    },
)
