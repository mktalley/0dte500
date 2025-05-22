import pytest
import main

class DummyOrder:
    def __init__(self, id, symbol):
        self.id = id
        self.symbol = symbol


def test_cancel_eod_success(monkeypatch):
    # Simulate two open orders
    orders = [DummyOrder('1', 'SPY'), DummyOrder('2', 'QQQ')]
    called = []
    logs = []
    alerts = []

    # Monkeypatch trade_client behavior
    monkeypatch.setattr(main.trade_client, 'get_orders', lambda: orders)
    monkeypatch.setattr(main.trade_client, 'cancel_order_by_id', lambda oid: called.append(oid))

    # Capture logs and alerts
    monkeypatch.setattr(main, 'log', lambda msg: logs.append(msg))
    monkeypatch.setattr(main, 'send_alert', lambda subject, body: alerts.append((subject, body)))

    # Ensure DRY_RUN is False
    main.DRY_RUN = False
    
    # Execute EOD cancel
    main.cancel_eod()

    # Assert cancellation calls
    assert called == ['1', '2']
    # Assert logs for each symbol
    assert logs == ['Cancelled EOD SPY', 'Cancelled EOD QQQ']
    # Assert alerts for each cancellation
    assert alerts == [('EOD Cancellation', 'SPY'), ('EOD Cancellation', 'QQQ')]


def test_cancel_eod_failure(monkeypatch):
    # Simulate one open order that fails cancellation
    orders = [DummyOrder('3', 'IWM')]
    logs = []
    alerts = []

    monkeypatch.setattr(main.trade_client, 'get_orders', lambda: orders)
    # Raise exception on cancel
    def raise_error(_):
        raise Exception('test failure')
    monkeypatch.setattr(main.trade_client, 'cancel_order_by_id', raise_error)

    monkeypatch.setattr(main, 'log', lambda msg: logs.append(msg))
    monkeypatch.setattr(main, 'send_alert', lambda subject, body: alerts.append((subject, body)))

    main.DRY_RUN = False
    main.cancel_eod()

    # On failure, no alerts, only a failure log
    assert any('Cancel failed: test failure' in msg for msg in logs)
    assert alerts == []
