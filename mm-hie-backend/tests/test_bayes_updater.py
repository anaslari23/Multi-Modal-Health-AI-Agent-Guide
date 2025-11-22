from __future__ import annotations

from app.fusion.bayes_updater import BayesianUpdater


def test_normalise_basic():
    b = BayesianUpdater()
    out = b.normalise({"A": 1.0, "B": 3.0})
    assert set(out.keys()) == {"A", "B"}
    assert abs(out["A"] - 0.25) < 1e-6
    assert abs(out["B"] - 0.75) < 1e-6


def test_normalise_handles_zero_or_negative():
    b = BayesianUpdater()
    out = b.normalise({"A": -1.0, "B": 0.0})
    assert out == {}


def test_update_single_likelihood():
    b = BayesianUpdater()
    prior = {"Pneumonia": 0.6, "Sepsis": 0.4}
    like = {"Pneumonia": 0.5, "Sepsis": 0.25}

    post = b.update(prior, like)

    # Unnormalised: Pn = 0.3, Se = 0.1 -> normalised: 0.75 / 0.25
    assert set(post.keys()) == {"Pneumonia", "Sepsis"}
    assert abs(post["Pneumonia"] - 0.75) < 1e-6
    assert abs(post["Sepsis"] - 0.25) < 1e-6


def test_update_missing_disease_is_neutral():
    b = BayesianUpdater()
    prior = {"Pneumonia": 0.6, "Sepsis": 0.4}
    like = {"Pneumonia": 0.5}  # no entry for Sepsis -> neutral (1.0)

    post = b.update(prior, like)

    # Unnormalised: Pn = 0.3, Se = 0.4 -> normalised.
    total = 0.3 + 0.4
    assert abs(post["Pneumonia"] - 0.3 / total) < 1e-6
    assert abs(post["Sepsis"] - 0.4 / total) < 1e-6


def test_update_multiple_likelihoods():
    b = BayesianUpdater()
    prior = {"A": 0.5, "B": 0.5}
    like1 = {"A": 0.2, "B": 0.8}
    like2 = {"A": 0.5, "B": 0.5}

    post = b.update(prior, like1, like2)

    # After like1: A=0.1, B=0.4 -> 0.2, 0.8
    # After like2: A=0.1, B=0.4 again -> 0.2, 0.8
    assert abs(post["A"] - 0.2) < 1e-6
    assert abs(post["B"] - 0.8) < 1e-6


def test_update_empty_prior_returns_empty():
    b = BayesianUpdater()
    assert b.update({}) == {}


def test_to_sorted_list_descending():
    b = BayesianUpdater()
    post = {"A": 0.1, "B": 0.7, "C": 0.2}
    out = b.to_sorted_list(post)

    assert [item["condition"] for item in out] == ["B", "C", "A"]
