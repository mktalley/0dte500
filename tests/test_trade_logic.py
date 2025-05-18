import pytest
import main
from datetime import datetime as real_datetime, time as real_time

class DummyAccount:
    def __init__(self, cash):
        self.cash = cash


def test_trade_early_exit_max_spreads(monkeypatch):
    # Setup dummy account with $200 -> cash//100 =2, MAX_OPEN_SPREADS=5 so max_spreads=2
    dummy_acc = DummyAccount('200')
    monkeypatch.setattr(main.trade_client, 'get_account', lambda: dummy_acc)
    # count_open_spreads returns 2, so 2>=2 triggers early exit
    monkeypatch.setattr(main, 'count_open_spreads', lambda: 2)
    # Now ensure trade() doesn't proceed to get options
    def fail_get_opts(req):
        pytest.fail("get_option_contracts should not be called")
    monkeypatch.setattr(main.trade_client, 'get_option_contracts', fail_get_opts)
    # Also ensure now < CANCEL_TIME
    class DummyDatetime:
        @classmethod
        def now(cls, tz):
            return real_datetime(2025,1,1,10,0,0, tzinfo=tz)
    monkeypatch.setattr(main, 'datetime', DummyDatetime)
    # Execute trade should not raise
    main.trade('SPY', 400)


def test_trade_after_cutoff(monkeypatch):
    # Setup dummy account with high cash
    dummy_acc = DummyAccount('1000')
    monkeypatch.setattr(main.trade_client, 'get_account', lambda: dummy_acc)
    # count_open_spreads returns 0 < max_spreads=5
    monkeypatch.setattr(main, 'count_open_spreads', lambda: 0)
    # datetime now returns time after CANCEL_TIME
    class DummyDatetime:
        @classmethod
        def now(cls, tz):
            return real_datetime(2025,1,1,16,0,0, tzinfo=tz)
    monkeypatch.setattr(main, 'datetime', DummyDatetime)
    # ensure get_option_contracts not called after cutoff
    monkeypatch.setattr(main.trade_client, 'get_option_contracts', lambda req: pytest.fail("Should not call get_option_contracts after cutoff"))
    # Execute trade should not raise
    main.trade('SPY', 400)


def test_max_spreads_calculation():
    assert min(main.MAX_OPEN_SPREADS, 50//main.MAX_RISK_PER_TRADE) == 0
    assert min(main.MAX_OPEN_SPREADS, 150//main.MAX_RISK_PER_TRADE) == 1
    assert min(main.MAX_OPEN_SPREADS, 450//main.MAX_RISK_PER_TRADE) == 4
    assert min(main.MAX_OPEN_SPREADS, 500//main.MAX_RISK_PER_TRADE) == 5
    assert min(main.MAX_OPEN_SPREADS, 1000//main.MAX_RISK_PER_TRADE) == 5
