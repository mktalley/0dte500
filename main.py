#!/usr/bin/env python3
# 0DTE Trading Bot (End-of-Day expiration, $500 Capital)

import os
import csv
import time
import argparse
import requests
import numpy as np
from datetime import datetime, date, time as dt_time
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
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

CAPITAL_POOL = 500
MAX_RISK_PER_TRADE = 100
MAX_OPEN_SPREADS = max(1, CAPITAL_POOL // MAX_RISK_PER_TRADE)  # Max spreads based on capital
MIN_CREDIT_PCT = 0.1
OI_THRESHOLD = 300
SHORT_DELTA_RANGE = (-0.45, -0.35)
LONG_DELTA_RANGE = (-0.25, -0.15)
STRIKE_RANGE = 0.1
SCAN_INTERVAL = 600  # seconds between scans
TIMEZONE = ZoneInfo("America/New_York")
CANCEL_TIME = dt_time(15, 55)
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Static tickers
TICKERS = ["SPY", "QQQ", "IWM"]  # Focus on liquid ETFs to maximize liquidity and profit potential

# CLI args
parser = argparse.ArgumentParser("0DTE Trading Bot")
parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Dry-run: no orders or cancels")
args = parser.parse_args()
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
    if SLACK_WEBHOOK_URL:
        try:
            requests.post(SLACK_WEBHOOK_URL, json={"text": f"{subject}\n{body}"})
        except Exception as e:
            log(f"Slack alert failed: {e}")
    else:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = os.getenv("EMAIL_ADDRESS")
            msg["To"] = os.getenv("TO_EMAIL")
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login(os.getenv("EMAIL_ADDRESS"), os.getenv("EMAIL_PASSWORD"))
            server.sendmail(msg["From"], [msg["To"]], msg.as_string())
            server.quit()
        except Exception as e:
            log(f"Email alert failed: {e}")


def calculate_iv(price, S, K, T, r, opt_type):
    intrinsic = max(0, (S - K) if opt_type == "call" else (K - S))
    if price <= intrinsic:
        return 0.0
    def f(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) - price
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) - price
    try:
        return brentq(f, 1e-6, 5.0)
    except:
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

# Fetch 0DTE options
def get_0dte_options(sym):
    spot = get_prices([sym]).get(sym)
    if not spot:
        return []
    lo, hi = str(spot * (1 - STRIKE_RANGE)), str(spot * (1 + STRIKE_RANGE))
    today = date.today()
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
    except:
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
    if count_open_spreads() >= MAX_OPEN_SPREADS:
        log(f"Max open spreads reached, skipping {symbol}")
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
    if credit < MIN_CREDIT_PCT * width or width * 100 > MAX_RISK_PER_TRADE:
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
