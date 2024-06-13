from setuptools import find_packages, setup

setup(
    name="alter_morph",
    version="0.0",
    description="Modelling altermagnets on amorphous lattices",
    long_description="",
    author="Peru D'Ornellas, Valentin Leeb",
    author_email="peru.dornellas@gmail.com",
    license="Apache Software License",
    home_page="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.2",
        "scipy",
        "matplotlib",
        "flake8",
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "nbmake",
        "pytest-github-actions-annotate-failures",
        "mpire",
    ],
)
