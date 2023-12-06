.. -*- mode: rst -*-

|Build yes|

.. |Build yes| image:: https://img.shields.io/badge/build-passing-<COLOR>.svg
   :target: https://github.com/a19ferna/equipy/actions/workflows/build-package.yml


**EquiPy** is a Python package implementing sequential fairness on the predicted outputs of Machine Learning models, when dealing with multiple sensitive attributes. This post-processing method progressively achieve fairness accross a set of sensitive features by leveraging multi-marginal Wasserstein barycenters, which extends the standard notion of Strong Demographic Parity to the case with multiple sensitive characteristics. This approach seamlessly extends
to approximate fairness, enveloping a framework accommodating the trade-off between performance and unfairness.

The project was started in 2023 by François Hu, Philipp Ratz, Suzie Grondin, Agathe Fernandes Machado and Arthur Charpentier, following the release of this paper "A Sequentially Fair Mechanism for Multiple Sensitive Attributes" (https://arxiv.org/pdf/2309.06627.pdf), written by François Hu, Philipp Ratz and Arthur Charpentier.  

Installation
------------

Dependencies
~~~~~~~~~~~~

EquiPy requires:

- Numpy (>= 1.17.3)
- Scipy (>= 1.5.0)
- Scikit-learn (== 1.3.0)
- Matplotlib (== 3.7.2)
- Pandas (== 2.0.3)
- Statsmodels (== 0.14.0)
- Seaborn (== 0.12.2)

User installation
~~~~~~~~~~~~~~~~~


To install EquiPy, use ``pip``::

    pip install equipy

Visualization
-------------

This package contains the module **graphs** which allows visualization of the resulting sequential fairness applied to a response variable.

Help and Support
----------------

Communication
~~~~~~~~~~~~~

Mailing list:

- hu.faugon@gmail.com
- suzie.grondin@gmail.com
- ratz.philipp@courrier.uqam.ca
- fernandes_machado.agathe@courrier.uqam.ca
- arthur.charpentier@gmail.com


