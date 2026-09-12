import pytest

from portfolio_engine.projection import project_wealth_path, wealth_scenarios


def test_project_wealth_path_no_contribution_matches_compound_interest():
    path = project_wealth_path(10000, 0, 0.10, 5)
    assert path[0] == pytest.approx(10000)
    assert path[5] == pytest.approx(10000 * 1.10**5, rel=1e-6)


def test_project_wealth_path_grows_with_contributions():
    no_contrib = project_wealth_path(10000, 0, 0.06, 10)
    with_contrib = project_wealth_path(10000, 500, 0.06, 10)
    assert with_contrib[10] > no_contrib[10]


def test_project_wealth_path_has_one_entry_per_year():
    path = project_wealth_path(5000, 100, 0.07, 20)
    assert list(path.index) == list(range(21))


def test_project_wealth_path_handles_severe_negative_return_without_crashing():
    path = project_wealth_path(10000, 100, -5.0, 5)  # absurd input, should clamp
    assert path[5] >= 0


def test_wealth_scenarios_ordering_optimistic_ge_expected_ge_pessimistic():
    scenarios = wealth_scenarios(10000, 500, expected_return=0.07, volatility=0.15, years=15)
    assert scenarios["optimistic"][15] >= scenarios["expected"][15]
    assert scenarios["expected"][15] >= scenarios["pessimistic"][15]


def test_wealth_scenarios_keys():
    scenarios = wealth_scenarios(1000, 0, 0.05, 0.10, 10)
    assert set(scenarios.keys()) == {"expected", "optimistic", "pessimistic"}
