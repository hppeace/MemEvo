from memevo.utils.models import TokenUsageLedger


def test_usage_ledger_keeps_empty_stages():
    ledger = TokenUsageLedger()

    with ledger.stage("retrieve"):
        pass

    assert ledger.summary()["stages"]["retrieve"]["total_tokens"] == 0
