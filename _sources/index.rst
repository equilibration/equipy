.. EquiPy documentation master file, created by
   sphinx-quickstart on Thu Dec  7 17:47:45 2023.

   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EquiPy's documentation!
==================================

**EquiPy** is a library implementing sequential fairness on the predicted outputs of Machine Learning models, when dealing with multiple sensitive attributes.
The library contains a set of classes that implement the methodology and a module for visualizations.

Under the hood, we use a post-processing method to progressively achieve fairness accross a set of sensitive features by leveraging multi-marginal Wasserstein barycenters, which extends the standard notion of Strong Demographic Parity to the case with multiple sensitive characteristics. This approach seamlessly extends to approximate fairness, enveloping a framework accommodating the trade-off between performance and unfairness. You can find the technical details in the technical paper https://arxiv.org/abs/2309.06627 (forthcoming at AAAI 2024).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
