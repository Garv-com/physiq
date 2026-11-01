from src.limits import HourlyBudget, ai_limit_per_hour, demo_mode


def test_demo_mode_is_off_by_default_and_reads_common_truthy_values(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert demo_mode() is False
    for value in ["true", "TRUE", "1", "yes", " True "]:
        monkeypatch.setenv("DEMO_MODE", value)
        assert demo_mode() is True
    monkeypatch.setenv("DEMO_MODE", "false")
    assert demo_mode() is False


def test_budget_blocks_after_the_limit_is_reached():
    budget = HourlyBudget()
    assert budget.try_spend(2, now=0)
    assert budget.try_spend(2, now=10)
    assert not budget.try_spend(2, now=20)


def test_budget_frees_up_after_an_hour():
    budget = HourlyBudget()
    budget.try_spend(1, now=0)
    assert not budget.try_spend(1, now=3599)
    assert budget.try_spend(1, now=3600)


def test_rejected_calls_do_not_use_up_budget():
    budget = HourlyBudget()
    budget.try_spend(1, now=0)
    for second in range(1, 50):
        assert not budget.try_spend(1, now=second)
    assert budget.try_spend(1, now=3600)


def test_limit_comes_from_the_environment_with_a_safe_fallback(monkeypatch):
    monkeypatch.setenv("AI_REQUESTS_PER_HOUR", "7")
    assert ai_limit_per_hour() == 7
    monkeypatch.setenv("AI_REQUESTS_PER_HOUR", "not a number")
    assert ai_limit_per_hour() == 30
    monkeypatch.setenv("AI_REQUESTS_PER_HOUR", "0")
    assert ai_limit_per_hour() == 1