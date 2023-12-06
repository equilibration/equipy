from setuptools import setup, find_packages

VERSION = "0.0.3-alpha-dev" 
DESCRIPTION = "Equipy is a tool for fast, online fairness calibration"

with open("README.rst", "r") as f:
    long_description = f.read()
LONG_DESCRIPTION = long_description

setup(
        name="equipy", 
        version=VERSION,
        author="Agathe F, Suzie G, Francois H, Philipp R",
        author_email="nocontact@email.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(include=["equipy", "equipy.*"]),
        install_requires=["numpy", "scipy", "scikit-learn",
                          "matplotlib", "pandas", "statsmodels",
                          "seaborn", "POT"], 
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