from ._wasserstein import FairWasserstein, MultiWasserstein

__all__ = ['FairWasserstein', 'MultiWasserstein']
#__all__ = ['fit', 'transform']

# Alias the fit methods
#fit = MultiWasserstein.fit
#transform = MultiWasserstein.transform