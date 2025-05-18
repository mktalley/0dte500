#!/usr/bin/env python3
# 0DTE Trading Bot (End-of-Day expiration, $500 Capital)

import os
import csv
import time
import argparse
import numpy as np
from datetime import datetime, date, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from scipy.stats import norm
from scipy.optimize import brentq

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce, AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest, OptionLegRequest, LimitOrderRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import OptionLatestQuoteRequest

# Configuration
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = False  # LIVE trading mode

# Risk and capital settings
MAX_RISK_PER_TRADE = 100  # $ risk per spread
MAX_OPEN_SPREADS = 5  # Hard cap on concurrent spreads based on $500 capital

# Dynamic sizing will respect cash balance (cash // MAX_RISK_PER_TRADE) up to MAX_OPEN_SPREADS
MIN_CREDIT_PCT = 0.1  # Minimum credit as a percentage of spread width
ABS_MIN_CREDIT = 0.1  # Absolute minimum credit per spread
OI_THRESHOLD = 300
SHORT_DELTA_RANGE = (-0.45, -0.35)
LONG_DELTA_RANGE = (-0.25, -0.15)
STRIKE_RANGE = 0.1
SCAN_INTERVAL = 600  # seconds between scans
TIMEZONE = ZoneInfo("America/New_York")
# Stop submissions at Alpaca's cutoff
CANCEL_TIME = dt_time(15, 30)  # 3:30 PM EST

# Static tickers
TICKERS = ["SPY", "QQQ", "DIA", "IWM"]

# CLI args
parser = argparse.ArgumentParser("0DTE Trading Bot")
parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Dry-run: no orders or cancels")
args, unknown = parser.parse_known_args()
DRY_RUN = args.dry_run

# Alpaca clients
trade_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
option_data_client = OptionHistoricalDataClient(API_KEY, API_SECRET)
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Daily log files
os.makedirs("logs", exist_ok=True)
today_str = date.today().isoformat()
TRADE_LOG = f"logs/trade_log_{today_str}.csv"
OPEN_LOG = f"logs/open_trades_{today_str}.csv"

# Initialize logs
if not os.path.exists(TRADE_LOG):
    with open(TRADE_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "symbol", "short_strike", "long_strike", "credit", "width", "status"
        ])
if not os.path.exists(OPEN_LOG):
    with open(OPEN_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "symbol", "short_strike", "long_strike", "credit", "width", "timestamp"
        ])

# Logging and alerts
def log(msg):
    print(f"[{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def send_alert(subject, body):
    # Simple alert - log only
    log(f"{subject}: {body}")


def calculate_iv(price, S, K, T, r, opt_type):
    intrinsic = max(0, (S - K) if opt_type == "call" else (K - S))
    if price <= intrinsic:
        return 0.0

    def f(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - price

    try:
        return brentq(f, 1e-6, 5.0)
    except Exception:
        return None


def calculate_delta(price, K, expiry, spot, r, opt_type):
    T = max((expiry - datetime.now(TIMEZONE)).total_seconds() / (365 * 24 * 3600), 1e-6)
    iv = calculate_iv(price, spot, K, T, r, opt_type)
    if iv is None:
        return None
    d1 = (np.log(spot / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return norm.cdf(d1) if opt_type == "call" else -norm.cdf(-d1)

# Market hours check
def is_market_open():
    now = datetime.now(TIMEZONE)
    return now.weekday() < 5 and dt_time(9, 30) <= now.time() <= dt_time(16, 0)

# Fetch underlying prices
def get_prices(tickers):
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=tickers)
        res = stock_data_client.get_stock_latest_trade(req)
        return {s: res[s].price for s in tickers if s in res}
    except Exception as e:
        log(f"Price fetch failed: {e}")
        return {}

# Fetch same-day 0DTE options only
def get_0dte_options(sym):
    spot = get_prices([sym]).get(sym)
    if not spot:
        return []
    lo, hi = str(spot * (1 - STRIKE_RANGE)), str(spot * (1 + STRIKE_RANGE))
    today = date.today()
    # First attempt: same-day expiry
    req = GetOptionContractsRequest(
        underlying_symbols=[sym],
        strike_price_gte=lo,
        strike_price_lte=hi,
        expiration_date=today,
        status=AssetStatus.ACTIVE,
        root_symbol=sym,
        type=ContractType.PUT
    )
    opts = trade_client.get_option_contracts(req).option_contracts
    if len(opts) < 5:
        time.sleep(1)
        opts = trade_client.get_option_contracts(req).option_contracts
    return opts

# Track open spreads
def count_open_spreads():
    try:
        with open(OPEN_LOG, newline="") as f:
            return len(list(csv.reader(f))) - 1
    except Exception:
        return 0


def log_open(symbol, short_str, long_str, credit, width):
    with open(OPEN_LOG, "a", newline="") as f:
            csv.writer(f).writerow([symbol, short_str, long_str, credit, width, datetime.now(TIMEZONE).isoformat()])

# End-of-day cancel
def cancel_eod():
    orders = trade_client.get_orders()
    for o in orders:
        try:
            if not DRY_RUN:
                trade_client.cancel_order_by_id(o.id)
            with open(TRADE_LOG, "a", newline="") as f:
                csv.writer(f).writerow([datetime.now(TIMEZONE).isoformat(), o.symbol, "", "", "", "", "cancelled_eod"])
            log(f"Cancelled EOD {o.symbol}")
            send_alert("EOD Cancellation", o.symbol)
        except Exception as e:
            log(f"Cancel failed: {e}")

# Execute trade
def trade(symbol, spot):
    now = datetime.now(TIMEZONE)
    # Early exit if after cutoff time
    if now.time() >= CANCEL_TIME:
        return
    # Enforce dynamic max spreads based on cash balance
    acct = trade_client.get_account()
    cash = float(acct.cash)
    max_spreads = min(MAX_OPEN_SPREADS, int(cash // MAX_RISK_PER_TRADE))
    current_spreads = count_open_spreads()
    if current_spreads >= max_spreads:
        log(f"Max open spreads reached ({current_spreads}/{max_spreads}), skipping {symbol}")
        return
    opts = get_0dte_options(symbol)
    short_put = long_put = None
    for o in opts:
        if int(o.open_interest or 0) < OI_THRESHOLD:
            continue
        q = option_data_client.get_option_latest_quote(OptionLatestQuoteRequest(symbol_or_symbols=[o.symbol])).get(o.symbol)
        if not q or not q.bid_price or not q.ask_price:
            continue
        price = (q.bid_price + q.ask_price) / 2
        expiry = datetime.combine(o.expiration_date, dt_time(16, 0)).replace(tzinfo=TIMEZONE)
        d = calculate_delta(price, float(o.strike_price), expiry, spot, 0.01, "put")
        # log each option check
        log(f"[{symbol}] Checking option: {o.symbol}, Price: {price:.6f}, Delta: {d}, OI: {int(o.open_interest or 0)}")
        if d and SHORT_DELTA_RANGE[0] <= d <= SHORT_DELTA_RANGE[1]:
            short_put = (o, price)
        if d and LONG_DELTA_RANGE[0] <= d <= LONG_DELTA_RANGE[1]:
            long_put = (o, price)
        if short_put and long_put:
            break
    if not short_put or not long_put:
        return
    credit = short_put[1] - long_put[1]
    width = abs(float(short_put[0].strike_price) - float(long_put[0].strike_price))
    min_credit = max(MIN_CREDIT_PCT * width, ABS_MIN_CREDIT)
    if credit < min_credit or width * 100 > MAX_RISK_PER_TRADE:
        return
    order = LimitOrderRequest(
        qty=1,
        limit_price=-round(credit, 2),
        order_class=OrderClass.MLEG,
        time_in_force=TimeInForce.DAY,
        legs=[
            OptionLegRequest(symbol=short_put[0].symbol, side=OrderSide.SELL, ratio_qty=1),
            OptionLegRequest(symbol=long_put[0].symbol, side=OrderSide.BUY, ratio_qty=1)
        ]
    )
    try:
        if not DRY_RUN:
            trade_client.submit_order(order)
            log_open(symbol, short_put[0].strike_price, long_put[0].strike_price, credit, width)
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(TIMEZONE).isoformat(), symbol,
                                     short_put[0].strike_price, long_put[0].strike_price,
                                     credit, width, "submitted" if not DRY_RUN else "dry_run"])
        send_alert("Trade Submitted", f"{symbol}: credit={credit:.2f}, width={width:.2f}")
    except Exception as e:
        log(f"Order failed: {e}")
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now(TIMEZONE).isoformat(), symbol,
                                     short_put[0].strike_price, long_put[0].strike_price,
                                     credit, width, "failed"])
        send_alert("Order Failed", f"{symbol}: {e}")

# Main loop
def main():
    log("🟢 Bot started")
    while True:
        now = datetime.now(TIMEZONE)
        if is_market_open():
            if now.time() >= CANCEL_TIME:
                cancel_eod()
            prices = get_prices(TICKERS)
            for sym, sp in prices.items():
                trade(sym, sp)
            log(f"Sleeping {SCAN_INTERVAL}s")
            time.sleep(SCAN_INTERVAL)
        else:
            log("Market closed, sleeping 15m")
            time.sleep(900)

if __name__ == "__main__":
    main()
