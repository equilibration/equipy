equipy.fairness package
=======================

Module contents
---------------

Main Classes to make predictions fair.

The module structure is as follows:

- The FairWasserstein base Class implements fairness adjustment related to a single sensitive attribute, using Wasserstein distance for both binary classification and regression tasks. In the case of binary classification, this class supports scores instead of classes. For more details, see E. Chzhen, C. Denis, M. Hebiri, L. Oneto and M. Pontil, "Fair Regression with Wasserstein Barycenters" (NeurIPS20).
- MultiWasserstein Class extends FairWasserstein for multi-sensitive attribute fairness adjustment in a sequential framework. For more details, see F. Hu, P. Ratz, A. Charpentier, "A Sequentially Fair Mechanism for Multiple Sensitive Attributes" (AAAI24).


.. autoclass:: equipy.fairness.FairWasserstein
   :members: fit, transform
   :exclude-members: modalities_calib, weights, ecdf, eqf
   :undoc-members:
   :show-inheritance:

.. autoclass:: equipy.fairness.MultiWasserstein
   :members: fit, transform
   :undoc-members:
   :show-inheritance:
