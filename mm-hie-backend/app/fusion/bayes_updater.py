from __future__ import annotations

from typing import Dict, Iterable, List


class BayesianUpdater:
    """Utility for Bayesian updating over discrete diseases.

    Inputs are simple dicts mapping disease name -> probability / likelihood.
    Probabilities are automatically normalised and zero-mass diseases are
    dropped from the posterior.
    """

    def normalise(self, probs: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(max(v, 0.0) for v in probs.values()))
        if total <= 0.0:
            # Avoid division by zero – return empty, caller must handle.
            return {}
        return {k: max(v, 0.0) / total for k, v in probs.items() if v > 0.0}

    def update(
        self,
        prior: Dict[str, float],
        *likelihoods: Iterable[Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute posterior given prior and one or more likelihood dicts.

        Any disease missing from a likelihood dict is assumed to have
        likelihood 1.0 for that evidence (i.e. neutral).
        """

        if not prior:
            return {}

        # Ensure prior is normalised.
        posterior: Dict[str, float] = self.normalise(prior)

        for like in likelihoods:
            like = dict(like) if like is not None else {}
            updated: Dict[str, float] = {}
            for disease, p in posterior.items():
                l = like.get(disease, 1.0)
                updated[disease] = p * max(l, 0.0)
            posterior = self.normalise(updated)
            if not posterior:
                break

        return posterior

    def to_sorted_list(self, posterior: Dict[str, float]) -> List[Dict[str, float]]:
        """Utility helper for returning a stable, sorted representation.

        Sorted descending by probability.
        """

        items = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
        return [{"condition": k, "prob": v} for k, v in items]
