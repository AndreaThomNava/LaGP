from setuptools import setup, find_packages

setup(
    name="lagp",  # Name of your package
    version="0.1",
    packages=find_packages(where="src"),  # Automatically finds the `lagp` package
    package_dir={"": "src"},  # Tells setuptools where to find the package
    install_requires=[  # List your project dependencies here
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "gpytorch",
        # Add other dependencies you are using
    ],
    python_requires=">=3.6",  # Adjust to your Python version
)

