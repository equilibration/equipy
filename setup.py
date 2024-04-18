from setuptools import setup, find_packages
import json 

with open("versioning/minor_build.json", "r") as f:
    minor_version = json.load(f)

with open("README.rst", "r") as f:
    long_description = f.read()

VERSION = f"0.0.{minor_version['buildNumber']}-alpha-dev" 
DESCRIPTION = "Equipy is a tool for fast, online fairness calibration"
LONG_DESCRIPTION = long_description

setup(
        name="equipy", 
        version=VERSION,
        author="Agathe F, Suzie G, Francois H, Philipp R, Arthur C",
        author_email="nocontact@email.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(include=["equipy", "equipy.*"]),
        install_requires=["numpy", "scipy", "scikit-learn",
                          "matplotlib", "pandas", "statsmodels",
                          "seaborn"], 
        setup_requires=["pytest-runner"],
        tests_require=["pytest"],
        keywords=["fairness", "wasserstein"],
        classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)