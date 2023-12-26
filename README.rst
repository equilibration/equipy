.. -*- mode: rst -*-

|Build yes|

.. |Build yes| image:: https://img.shields.io/badge/build-passing-<COLOR>.svg
   :target: https://github.com/a19ferna/equipy/actions/workflows/build-package.yml


**EquiPy** is a Python package implementing sequential fairness on the predicted outputs of Machine Learning models, when dealing with multiple sensitive attributes. This post-processing method progressively achieve fairness accross a set of sensitive features by leveraging multi-marginal Wasserstein barycenters, which extends the standard notion of Strong Demographic Parity to the case with multiple sensitive characteristics. This approach seamlessly extends
to approximate fairness, enveloping a framework accommodating the trade-off between performance and unfairness.

The project was started in 2023 by François Hu, Philipp Ratz, Suzie Grondin, Agathe Fernandes Machado and Arthur Charpentier, following the release of this paper "A Sequentially Fair Mechanism for Multiple Sensitive Attributes" (https://arxiv.org/pdf/2309.06627.pdf), written by François Hu, Philipp Ratz and Arthur Charpentier.

Website : https://equilibration.github.io/equipy/

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
- POT (==0.9.1)

User installation
~~~~~~~~~~~~~~~~~


To install EquiPy, use ``pip``::

    pip install equipy

Project Tree Structure
------------

The following is the tree structure of the project:

.. code-block:: plaintext

    equipy/
        ├── equipy/
        |   ├── __init__.py
        |   ├── fairness/
        |   │   ├── __init__.py
        |   |   ├── _base.py
        |   |   ├── _wasserstein.py
        |   ├── graphs/
        |   │   ├── __init__.py
        |   │   ├── _arrow_plot.py
        |   │   ├── _density_plot.py
        |   │   ├── _waterfall_plot.py
        |   ├── metrics/
        |   │   ├── __init__.py
        |   │   ├── _fairness_metrics.py
        |   │   ├── _performance_metrics.py
        |   ├── utils/
        |   │   ├── __init__.py
        |   │   ├── checkers.py
        |   │   ├── permutations/
        |   │   |   ├── __init__.py
        |   │   |   ├── _compute_permutations.py
        |   │   |   ├── metrics/
        |   │   |   |   ├── __init__.py
        |   │   |   |   ├── _fairness_permutations.py
        |   │   |   |   ├── _performance_permutations.py
        ├── .gitignore
        ├── LICENSE
        ├── README.rst
        ├── requirements.txt
        ├── setup.py
        └── tests.py


Visualization
-------------

This package contains the module **graphs** which allows visualization of the resulting sequential fairness applied to a response variable.

.. image:: https://github.com/equilibration/equipy/tree/feature-corrections/examples/images/arrow_plot_3_sa.png
  :width: 400
  :alt: (Risk, Unfairness) phase diagram that shows the sequential fairness approach for three sensitive features

![Alt text](https://github.com/equilibration/equipy/tree/feature-corrections/examples/images/arrow_plot_3_sa.png)

Help and Support
----------------

Communication
~~~~~~~~~~~~~

If you have any inquiries, feel free to contact us:

- François Hu : hu.faugon@gmail.com
- Suzie Grondin : suzie.grondin@gmail.com
- Philipp Ratz : ratz.philipp@courrier.uqam.ca
- Agathe Fernandes Machado : fernandes_machado.agathe@courrier.uqam.ca
- Arthur Charpentier : arthur.charpentier@gmail.com


